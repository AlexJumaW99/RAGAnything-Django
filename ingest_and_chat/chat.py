# ingest_and_chat/chat.py
"""
Chat and RAG query engine.

Provides:
  - Hybrid search (pgvector cosine + full-text GIN)
  - Pandas code generation + sandboxed execution for structured data
  - Conversation management with message history
  - Session-scoped queries
  - Multi-provider LLM support (Gemini, Claude, Ollama)

All functions are designed to be called from views.py.
"""

import io
import os
import sys
import json
import uuid
import threading
import traceback
import logging
from typing import Optional

from psycopg2.extras import Json, RealDictCursor

from .db import get_connection
from .config import (
    get_embedding_model,
    PG_HOST, PG_PORT, PG_USERNAME, PG_PASSWORD, PG_DATABASE,
    SIMILARITY_THRESHOLD,
    CHAT_HISTORY_LIMIT,
    PANDAS_EXEC_TIMEOUT,
    HYBRID_VECTOR_WEIGHT,
    HYBRID_TEXT_WEIGHT,
)

logger = logging.getLogger("ingest_and_chat")


# ---------------------------------------------------------------------------
# Database connection helper
# ---------------------------------------------------------------------------

def _get_conn():
    """
    Return a connection with DB + tables guaranteed to exist.
    Delegates to db.get_connection() which handles auto-initialization.
    """
    return get_connection()


# ---------------------------------------------------------------------------
# LLM provider helper
# ---------------------------------------------------------------------------

def _get_llm_provider(provider_name: Optional[str] = None):
    """
    Return an LLM provider instance.

    If provider_name is given, use that provider.
    Otherwise fall back to the Gemini provider (legacy behaviour).
    """
    if provider_name:
        from .llm_providers import get_provider
        return get_provider(provider_name)
    else:
        # Legacy fallback: use Gemini via the old config.get_llm() path
        from .config import get_llm
        llm = get_llm()
        # Wrap it in a simple adapter so .invoke() returns a string
        class _LangchainAdapter:
            def invoke(self, prompt):
                resp = llm.invoke(prompt)
                return resp.content
        return _LangchainAdapter()


# ===========================================================================
# Conversation Management
# ===========================================================================

def create_conversation(session_id: Optional[str] = None, title: Optional[str] = None) -> dict:
    """Create a new conversation, optionally scoped to an ingestion session."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("""
            INSERT INTO conversations (session_id, title)
            VALUES (%s, %s)
            RETURNING id, session_id, title, created_at, updated_at
        """, (session_id, title or "New conversation"))
        row = cur.fetchone()
        conn.commit()
        return dict(row)
    finally:
        cur.close()
        conn.close()


def list_conversations(session_id: Optional[str] = None, limit: int = 50) -> list:
    """List conversations, optionally filtered by session."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        if session_id:
            cur.execute("""
                SELECT c.*, COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN chat_messages m ON m.conversation_id = c.id
                WHERE c.session_id = %s
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT %s
            """, (session_id, limit))
        else:
            cur.execute("""
                SELECT c.*, COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN chat_messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT %s
            """, (limit,))
        return [dict(row) for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


def get_conversation_history(conversation_id: str, limit: Optional[int] = None) -> list:
    """Get all messages in a conversation, ordered chronologically."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        effective_limit = limit or CHAT_HISTORY_LIMIT
        cur.execute("""
            SELECT id, conversation_id, session_id, role, content,
                   sources, tool_calls, created_at
            FROM chat_messages
            WHERE conversation_id = %s
            ORDER BY created_at ASC
            LIMIT %s
        """, (conversation_id, effective_limit))
        return [dict(row) for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation and all its messages."""
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM chat_messages WHERE conversation_id = %s", (conversation_id,))
        cur.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        cur.close()
        conn.close()


def _save_message(cur, conversation_id, session_id, role, content,
                   sources=None, tool_calls=None):
    """Insert a chat message and update conversation timestamp."""
    cur.execute("""
        INSERT INTO chat_messages
            (conversation_id, session_id, role, content, sources, tool_calls)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id, created_at
    """, (
        conversation_id, session_id, role, content,
        Json(sources) if sources else None,
        Json(tool_calls) if tool_calls else None,
    ))
    msg_row = cur.fetchone()

    cur.execute("""
        UPDATE conversations SET updated_at = NOW() WHERE id = %s
    """, (conversation_id,))

    return {"id": str(msg_row[0]), "created_at": msg_row[1].isoformat()}


# ===========================================================================
# Hybrid Search
# ===========================================================================

def _hybrid_search_documents(cur, query_embedding, query_text,
                             session_id=None, limit=5):
    """
    Combine vector cosine similarity with full-text search relevance.
    Returns merged results ranked by weighted score.
    """
    formatted_emb = f"[{','.join(map(str, query_embedding))}]"
    ts_query = " & ".join(
        word for word in query_text.split()
        if len(word) > 2 and word.isalnum()
    )
    if not ts_query:
        ts_query = query_text.split()[0] if query_text.split() else "a"

    session_filter = ""
    params = [formatted_emb, formatted_emb]

    if session_id:
        session_filter = "AND session_id = %s"
        params.append(session_id)

    # Query combines vector score and text score
    sql = f"""
        WITH vector_scores AS (
            SELECT id, filepath, content, metadata,
                   1 - (embedding <=> %s::vector) AS vec_score
            FROM document_chunks
            WHERE embedding IS NOT NULL {session_filter}
            ORDER BY embedding <=> %s::vector
            LIMIT {limit * 2}
        ),
        text_scores AS (
            SELECT id,
                   ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS txt_score
            FROM document_chunks
            WHERE search_vector @@ plainto_tsquery('english', %s)
            {session_filter.replace('session_id', 'dc.session_id') if session_id else ''}
            LIMIT {limit * 2}
        )
        SELECT
            v.id, v.filepath, v.content, v.metadata, v.vec_score,
            COALESCE(t.txt_score, 0) AS txt_score,
            ({HYBRID_VECTOR_WEIGHT} * v.vec_score +
             {HYBRID_TEXT_WEIGHT} * COALESCE(t.txt_score, 0)) AS hybrid_score
        FROM vector_scores v
        LEFT JOIN text_scores t ON v.id = t.id
        ORDER BY hybrid_score DESC
        LIMIT %s
    """

    # Build params: vec search needs emb twice, text search needs query twice,
    # plus optional session_ids
    query_params = [formatted_emb, formatted_emb]
    if session_id:
        query_params.append(session_id)
    query_params.extend([query_text, query_text])
    if session_id:
        query_params.append(session_id)
    query_params.append(limit)

    cur.execute(sql, query_params)
    results = []
    for row in cur.fetchall():
        results.append({
            "source": row[1],
            "content": row[2],
            "metadata": row[3] if isinstance(row[3], dict) else {},
            "vec_score": round(float(row[4]), 4),
            "txt_score": round(float(row[5]), 4),
            "hybrid_score": round(float(row[6]), 4),
            "table": "document_chunks",
        })
    return results


def _search_media(cur, query_embedding, query_text, session_id=None, limit=3):
    """Vector + text search on media file transcripts."""
    formatted_emb = f"[{','.join(map(str, query_embedding))}]"

    session_clause = "AND session_id = %s" if session_id else ""
    params = [formatted_emb, formatted_emb]
    if session_id:
        params.append(session_id)
    params.append(limit)

    cur.execute(f"""
        SELECT filepath, transcript, metadata, storage_path,
               1 - (transcript_embedding <=> %s::vector) AS similarity
        FROM media_files
        WHERE transcript_embedding IS NOT NULL {session_clause}
        ORDER BY transcript_embedding <=> %s::vector
        LIMIT %s
    """, params)

    results = []
    for row in cur.fetchall():
        results.append({
            "source": row[0],
            "content": row[1] or "",
            "metadata": row[2] if isinstance(row[2], dict) else {},
            "storage_path": row[3],
            "similarity": round(float(row[4]), 4),
            "table": "media_files",
        })
    return results


def _search_structured(cur, query_embedding, query_text, session_id=None, limit=3):
    """Search structured files by schema embedding."""
    formatted_emb = f"[{','.join(map(str, query_embedding))}]"

    session_clause = "AND session_id = %s" if session_id else ""
    params = [formatted_emb, formatted_emb]
    if session_id:
        params.append(session_id)
    params.append(limit)

    cur.execute(f"""
        SELECT filepath, schema_description, sample_rows, column_names,
               dtypes, row_count, metadata, storage_path,
               1 - (summary_embedding <=> %s::vector) AS similarity
        FROM structured_files
        WHERE summary_embedding IS NOT NULL {session_clause}
        ORDER BY summary_embedding <=> %s::vector
        LIMIT %s
    """, params)

    results = []
    for row in cur.fetchall():
        results.append({
            "source": row[0],
            "schema_description": row[1] or "",
            "sample_rows": row[2] if isinstance(row[2], list) else [],
            "column_names": row[3] if isinstance(row[3], list) else [],
            "dtypes": row[4] if isinstance(row[4], dict) else {},
            "row_count": row[5],
            "metadata": row[6] if isinstance(row[6], dict) else {},
            "storage_path": row[7],
            "similarity": round(float(row[8]), 4),
            "table": "structured_files",
        })
    return results


# ===========================================================================
# Pandas Code Execution (sandboxed)
# ===========================================================================

def _generate_pandas_code(llm_provider, question: str, structured_result: dict) -> str:
    """Ask the LLM to write pandas code to answer a question about a dataset."""
    prompt = (
        "You are a Python data analyst. Write pandas code to answer the "
        "user's question about this dataset.\n\n"
        f"FILE PATH (use this exact path): {structured_result['storage_path']}\n"
        f"FILE TYPE: {structured_result.get('source', '').split('.')[-1]}\n\n"
        f"SCHEMA:\n{structured_result.get('schema_description', 'N/A')}\n\n"
        f"SAMPLE ROWS:\n{json.dumps(structured_result.get('sample_rows', []), indent=2)}\n\n"
        f"COLUMN TYPES:\n{json.dumps(structured_result.get('dtypes', {}), indent=2)}\n\n"
        f"USER QUESTION: {question}\n\n"
        "RULES:\n"
        "- Write ONLY executable Python code, no markdown fences, no explanations.\n"
        "- Import pandas as pd and numpy as np at the top.\n"
        "- Load the file using the FILE PATH variable `file_path` (already defined).\n"
        "- For CSV/TSV use pd.read_csv(file_path), for XLSX use pd.read_excel(file_path).\n"
        "- Store the final answer in a variable called `result`.\n"
        "- The `result` should be a string or a simple value (not a DataFrame).\n"
        "- If returning a DataFrame, convert it: result = df.to_string()\n"
        "- Handle edge cases (empty data, missing columns) gracefully.\n"
    )

    try:
        code = llm_provider.invoke(prompt).strip()
        # Strip markdown fences if the LLM included them despite instructions
        if code.startswith("```"):
            lines = code.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            code = "\n".join(lines)
        return code
    except Exception as e:
        logger.error("Failed to generate pandas code: %s", e)
        return None


def _execute_pandas_code(code: str, storage_path: str) -> dict:
    """
    Execute LLM-generated pandas code in a restricted environment.

    Uses a background thread with timeout for safety.
    """
    import pandas as pd
    import numpy as np

    result_holder = {"success": False, "result": None, "error": None}

    def _run():
        output_capture = io.StringIO()
        safe_builtins = {
            "print": lambda *args, **kwargs: output_capture.write(
                " ".join(str(a) for a in args) + "\n"
            ),
            "len": len, "range": range, "str": str, "int": int, "float": float,
            "bool": bool, "list": list, "dict": dict, "tuple": tuple, "set": set,
            "sorted": sorted, "min": min, "max": max, "sum": sum, "round": round,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
            "isinstance": isinstance, "type": type, "abs": abs, "any": any, "all": all,
            "True": True, "False": False, "None": None,
            "ValueError": ValueError, "TypeError": TypeError,
            "KeyError": KeyError, "IndexError": IndexError,
            "Exception": Exception,
        }

        exec_globals = {
            "__builtins__": safe_builtins,
            "pd": pd,
            "np": np,
            "file_path": storage_path,
        }
        exec_locals = {}

        try:
            exec(code, exec_globals, exec_locals)
            printed = output_capture.getvalue().strip()
            result = exec_locals.get("result", exec_locals.get("answer", printed))
            result_holder["success"] = True
            result_holder["result"] = str(result) if result is not None else printed
        except Exception as e:
            result_holder["success"] = False
            result_holder["error"] = f"{type(e).__name__}: {str(e)}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=PANDAS_EXEC_TIMEOUT)

    if thread.is_alive():
        return {
            "success": False,
            "error": f"Execution timed out after {PANDAS_EXEC_TIMEOUT}s",
            "code": code,
        }

    return {**result_holder, "code": code}


# ===========================================================================
# Main Chat Function
# ===========================================================================

def chat(conversation_id: str, question: str,
         session_id: Optional[str] = None,
         provider: Optional[str] = None) -> dict:
    """
    Full chat flow:
      1. Load conversation history
      2. Embed question -> hybrid search across all tables
      3. For structured hits, generate + execute pandas code
      4. Build prompt with context + history
      5. Get LLM answer (using requested provider)
      6. Save messages
      7. Return answer + sources + tool results

    Args:
        conversation_id: UUID of the conversation
        question:        User's question text
        session_id:      Optional ingestion session UUID to scope search
        provider:        LLM provider name (gemini, claude, ollama)

    Returns:
        dict with keys: answer, sources, tool_results, tables_searched,
                        user_message_id, assistant_message_id, provider_used
    """
    conn = _get_conn()
    cur = conn.cursor()

    try:
        # Tables are guaranteed to exist by db.get_connection() / ensure_db_ready().
        cur.execute("SELECT COUNT(*) FROM ingestion_sessions WHERE status = 'complete'")
        session_count = cur.fetchone()[0]

        if session_count == 0:
            cur.close()
            conn.close()
            return {
                "answer": None,
                "error": "No data has been ingested yet. Run an ingestion first.",
                "sources": [],
                "tables_searched": [],
            }

        # 1. Get conversation history for context
        history = []
        cur.execute("""
            SELECT role, content FROM chat_messages
            WHERE conversation_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (conversation_id, CHAT_HISTORY_LIMIT))
        rows = cur.fetchall()
        history = [{"role": r[0], "content": r[1]} for r in reversed(rows)]

        # 2. Embed the question
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode(question).tolist()

        # 3. Hybrid search across all tables
        retrieved_chunks = []
        structured_hits = []
        tables_searched = ["document_chunks", "media_files", "structured_files"]

        doc_results = _hybrid_search_documents(
            cur, query_embedding, question, session_id, limit=5
        )
        retrieved_chunks.extend(doc_results)

        media_results = _search_media(
            cur, query_embedding, question, session_id, limit=3
        )
        retrieved_chunks.extend(media_results)

        struct_results = _search_structured(
            cur, query_embedding, question, session_id, limit=3
        )
        structured_hits.extend(struct_results)

        # 4. Get the LLM provider
        llm_provider = _get_llm_provider(provider)
        provider_used = provider or "gemini"

        # Execute pandas code for relevant structured files
        tool_results = []

        for sf in structured_hits:
            if not sf.get("storage_path") or not os.path.exists(sf.get("storage_path", "")):
                retrieved_chunks.append({
                    "source": sf["source"],
                    "content": (
                        f"[Structured File]\n{sf.get('schema_description', '')}\n\n"
                        f"Sample rows:\n{json.dumps(sf.get('sample_rows', []), indent=2)}"
                    ),
                    "metadata": sf.get("metadata", {}),
                    "similarity": sf.get("similarity", 0),
                    "table": "structured_files",
                })
                continue

            code = _generate_pandas_code(llm_provider, question, sf)
            if not code:
                continue

            exec_result = _execute_pandas_code(code, sf["storage_path"])
            tool_results.append({
                "file": sf["source"],
                "storage_path": sf["storage_path"],
                **exec_result,
            })

            if exec_result["success"]:
                retrieved_chunks.append({
                    "source": sf["source"],
                    "content": (
                        f"[Pandas Analysis Result for {os.path.basename(sf['source'])}]\n"
                        f"Code executed:\n{exec_result['code']}\n\n"
                        f"Result:\n{exec_result['result']}"
                    ),
                    "metadata": sf.get("metadata", {}),
                    "similarity": sf.get("similarity", 0),
                    "table": "structured_files (pandas_exec)",
                })
            else:
                retrieved_chunks.append({
                    "source": sf["source"],
                    "content": (
                        f"[Structured File — code execution failed: {exec_result.get('error', 'unknown')}]\n"
                        f"Schema:\n{sf.get('schema_description', '')}\n\n"
                        f"Sample rows:\n{json.dumps(sf.get('sample_rows', []), indent=2)}"
                    ),
                    "metadata": sf.get("metadata", {}),
                    "similarity": sf.get("similarity", 0),
                    "table": "structured_files",
                })

        # 5. Build final prompt
        if not retrieved_chunks:
            answer = "No relevant results found in the knowledge base."
        else:
            def _score(c):
                return c.get("hybrid_score", c.get("similarity", 0))

            retrieved_chunks.sort(key=_score, reverse=True)

            context_parts = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                score = _score(chunk)
                context_parts.append(
                    f"[Source {i}] (relevance: {score}) — {chunk['source']}\n"
                    f"{chunk['content'][:2000]}"
                )
            context_block = "\n\n---\n\n".join(context_parts)

            history_block = ""
            if history:
                history_lines = []
                for msg in history[-10:]:
                    role_label = "User" if msg["role"] == "user" else "Assistant"
                    history_lines.append(f"{role_label}: {msg['content'][:500]}")
                history_block = (
                    "\n--- CONVERSATION HISTORY ---\n"
                    + "\n".join(history_lines)
                    + "\n"
                )

            rag_prompt = (
                "You are a helpful assistant answering questions ONLY based on "
                "the retrieved documents and data analysis results below.\n"
                "IMPORTANT: You must ONLY answer based on the provided context. "
                "If the context doesn't contain enough information to answer the "
                "question, clearly state that the information is not available in "
                "the knowledge base. Do NOT use your general knowledge.\n"
                "Cite which source(s) you used by referencing [Source N].\n"
                "If pandas analysis results are included, reference those findings.\n"
                f"{history_block}\n"
                f"--- RETRIEVED CONTEXT ---\n{context_block}\n\n"
                f"--- USER QUESTION ---\n{question}\n\n"
                "Provide a clear, concise answer based ONLY on the context above:"
            )

            try:
                answer = llm_provider.invoke(rag_prompt)
            except Exception as e:
                answer = f"LLM error: {str(e)}"

        # 6. Build sources list
        sources = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            sources.append({
                "index": i,
                "source": chunk["source"],
                "score": _score(chunk),
                "table": chunk["table"],
                "preview": (
                    (chunk["content"][:200] + "...")
                    if len(chunk["content"]) > 200
                    else chunk["content"]
                ),
            })

        # 7. Save messages to database
        user_msg_info = _save_message(
            cur, conversation_id, session_id, "user", question,
        )
        assistant_msg_info = _save_message(
            cur, conversation_id, session_id, "assistant", answer,
            sources=sources,
            tool_calls=tool_results if tool_results else None,
        )
        conn.commit()

        cur.close()
        conn.close()

        return {
            "answer": answer,
            "sources": sources,
            "tool_results": tool_results,
            "tables_searched": tables_searched,
            "user_message_id": user_msg_info["id"],
            "assistant_message_id": assistant_msg_info["id"],
            "provider_used": provider_used,
        }

    except Exception as e:
        logger.exception("Chat query failed")
        try:
            cur.close()
            conn.close()
        except Exception:
            pass
        raise


# ===========================================================================
# Session Management (list / get ingestion sessions)
# ===========================================================================

def list_sessions(limit: int = 50) -> list:
    """List all ingestion sessions."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("""
            SELECT id, name, target_path, status, file_count,
                   records_inserted, created_at, completed_at
            FROM ingestion_sessions
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        return [dict(row) for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


def get_session(session_id: str) -> dict:
    """Get details of a single ingestion session."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("""
            SELECT * FROM ingestion_sessions WHERE id = %s
        """, (session_id,))
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        cur.close()
        conn.close()


def delete_session(session_id: str) -> bool:
    """
    Delete an ingestion session and all associated data.
    CASCADE deletes handle document_chunks, media_files, structured_files.
    """
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT storage_path FROM media_files WHERE session_id = %s AND storage_path IS NOT NULL",
            (session_id,),
        )
        for row in cur.fetchall():
            if row[0] and os.path.exists(row[0]):
                try:
                    os.remove(row[0])
                except OSError:
                    pass

        cur.execute(
            "SELECT storage_path FROM structured_files WHERE session_id = %s AND storage_path IS NOT NULL",
            (session_id,),
        )
        for row in cur.fetchall():
            if row[0] and os.path.exists(row[0]):
                try:
                    os.remove(row[0])
                except OSError:
                    pass

        cur.execute("DELETE FROM ingestion_sessions WHERE id = %s", (session_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        cur.close()
        conn.close()
