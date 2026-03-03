# ingest_and_chat/services.py
"""
Business logic layer: ingestion orchestration and RAG query engine.

These are called by the Django views and keep the view layer thin.
"""

import os
import io
import sys
import uuid
import json
import datetime
import threading
import traceback
import logging

logger = logging.getLogger("ingest_and_chat")

# ---------------------------------------------------------------------------
# Ingestion tracking (module-level singletons)
# ---------------------------------------------------------------------------

_ingestion_lock = threading.Lock()
_ingestion_thread = None
_ingestion_cancel = threading.Event()


def is_ingestion_running():
    global _ingestion_thread
    return _ingestion_thread is not None and _ingestion_thread.is_alive()


def cancel_ingestion():
    """Signal the running ingestion to stop."""
    global _ingestion_thread
    _ingestion_cancel.set()
    was_running = is_ingestion_running()
    return was_running


# ---------------------------------------------------------------------------
# Output directory creation
# ---------------------------------------------------------------------------

def create_output_dir(target_path):
    """Creates a timestamped output directory under PROCESSED_OUTPUT_ROOT."""
    from .config import PROCESSED_OUTPUT_ROOT

    project_name = os.path.basename(os.path.abspath(target_path))
    project_name = project_name.replace(" ", "_").replace(".", "_")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(PROCESSED_OUTPUT_ROOT, f"{project_name}_{timestamp}")

    subdirs = [
        "code", "text", "config", "pdf", "office",
        "media/images", "media/audio",
        "structured", "metadata",
    ]
    for sub in subdirs:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    return run_dir


# ---------------------------------------------------------------------------
# Ingestion (streaming)
# ---------------------------------------------------------------------------

def run_ingestion_stream(target_path):
    """
    Generator that runs the full RAG ingestion pipeline and yields
    SSE-formatted JSON events.

    Yields: strings of the form  'data: {"event": "...", "data": {...}}\n'
    """
    import queue

    abs_target = os.path.abspath(target_path)
    event_queue = queue.Queue()
    _ingestion_cancel.clear()

    def emit(event, data):
        event_queue.put(json.dumps({"event": event, "data": data}) + "\n")

    def _pipeline_worker():
        global _ingestion_thread
        try:
            emit("status", {"message": "Initializing pipeline...", "step": "init"})

            from .pipeline_graph import create_ingestion_graph
            from .config import PG_HOST, PG_PORT, PG_USERNAME, PG_PASSWORD, PG_DATABASE
            from .utils import save_graph_image

            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            emit("status", {
                "message": "Creating ingestion graph...",
                "step": "graph_init",
                "session_id": thread_id,
            })

            app_graph = create_ingestion_graph()

            try:
                save_graph_image(app_graph, thread_id)
            except Exception:
                pass

            output_dir = create_output_dir(abs_target)

            emit("status", {
                "message": f"Output directory: {output_dir}",
                "step": "output_dir",
                "output_dir": output_dir,
            })

            initial_state = {
                "session_id": thread_id,
                "target_path": abs_target,
                "project_tree": None,
                "output_dir": output_dir,
                "classified_files": None,
                "processed_documents": None,
                "processed_media": None,
                "processed_structured": None,
                "file_metadata": None,
                "pg_host": PG_HOST,
                "pg_port": PG_PORT,
                "pg_username": PG_USERNAME,
                "pg_password": PG_PASSWORD,
                "pg_database": PG_DATABASE,
                "records_inserted": 0,
                "current_step": None,
                "steps_completed": [],
                "has_error": False,
                "errors": [],
                "error_log_path": None,
                "debug_summary": None,
                "last_command": None,
                "last_stdout": None,
                "last_stderr": None,
            }

            emit("status", {"message": "Starting ingestion pipeline...", "step": "pipeline_start"})

            # Capture stdout from nodes so we can relay log lines
            old_stdout = sys.stdout
            captured = io.StringIO()

            class TeeWriter:
                def __init__(self, original, capture, emitter):
                    self.original = original
                    self.capture = capture
                    self.emitter = emitter
                def write(self, text):
                    self.original.write(text)
                    self.capture.write(text)
                    if text.strip():
                        self.emitter("log", {"text": text.rstrip()})
                def flush(self):
                    self.original.flush()
                    self.capture.flush()

            sys.stdout = TeeWriter(old_stdout, captured, emit)

            try:
                final_state = app_graph.invoke(initial_state, config)
            finally:
                sys.stdout = old_stdout

            if final_state.get("has_error", False):
                emit("error", {
                    "message": "Ingestion failed",
                    "step": final_state.get("current_step", "unknown"),
                    "errors": final_state.get("errors", []),
                    "error_log_path": final_state.get("error_log_path"),
                })
            else:
                classified = final_state.get("classified_files") or {}
                file_counts = {cat: len(files) for cat, files in classified.items() if files}
                emit("complete", {
                    "message": "Ingestion completed successfully",
                    "session_id": thread_id,
                    "records_inserted": final_state.get("records_inserted", 0),
                    "steps_completed": final_state.get("steps_completed", []),
                    "file_counts": file_counts,
                    "output_dir": output_dir,
                })

        except Exception as e:
            emit("error", {
                "message": f"Pipeline exception: {str(e)}",
                "traceback": traceback.format_exc(),
            })
        finally:
            emit("done", {"message": "Stream finished"})
            _ingestion_thread = None

    # Start pipeline in background thread
    global _ingestion_thread
    with _ingestion_lock:
        if is_ingestion_running():
            yield json.dumps({
                "event": "error",
                "data": {"message": "An ingestion is already running."},
            }) + "\n"
            return
        _ingestion_thread = threading.Thread(target=_pipeline_worker, daemon=True)
        _ingestion_thread.start()

    # Drain the event queue and yield SSE lines
    while True:
        try:
            msg = event_queue.get(timeout=120)
            yield f"data: {msg}\n"
            parsed = json.loads(msg)
            if parsed.get("event") == "done":
                break
        except queue.Empty:
            yield f"data: {json.dumps({'event': 'keepalive', 'data': {}})}\n"


# ---------------------------------------------------------------------------
# RAG Query
# ---------------------------------------------------------------------------

def query_rag(question):
    """
    Embed question, search pgvector, ask LLM, return answer + sources.

    Returns a dict:
        {"answer": str, "sources": list[dict], "tables_searched": list[str]}
    """
    import psycopg2
    from .config import (
        get_llm, get_embedding_model,
        PG_HOST, PG_PORT, PG_USERNAME, PG_PASSWORD, PG_DATABASE,
    )

    llm = get_llm()
    embedding_model = get_embedding_model()

    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USERNAME,
        password=PG_PASSWORD, dbname=PG_DATABASE, connect_timeout=10,
    )

    # Check which tables exist
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name IN ('document_chunks', 'media_files', 'structured_files')
    """)
    existing_tables = {row[0] for row in cur.fetchall()}
    cur.close()

    if not existing_tables:
        conn.close()
        return {
            "answer": None,
            "error": "No RAG tables found. Run an ingestion first.",
            "sources": [],
            "tables_searched": [],
        }

    # Embed the query
    query_embedding = embedding_model.encode(question).tolist()
    formatted_emb = f"[{','.join(map(str, query_embedding))}]"

    retrieved_chunks = []
    cur = conn.cursor()

    if "document_chunks" in existing_tables:
        cur.execute("""
            SELECT filepath, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT 5
        """, (formatted_emb, formatted_emb))
        for row in cur.fetchall():
            retrieved_chunks.append({
                "source": row[0], "content": row[1],
                "metadata": row[2] if isinstance(row[2], dict) else {},
                "similarity": round(float(row[3]), 4),
                "table": "document_chunks",
            })

    if "media_files" in existing_tables:
        cur.execute("""
            SELECT filepath, transcript, metadata,
                   1 - (transcript_embedding <=> %s::vector) AS similarity
            FROM media_files
            WHERE transcript_embedding IS NOT NULL
            ORDER BY transcript_embedding <=> %s::vector
            LIMIT 3
        """, (formatted_emb, formatted_emb))
        for row in cur.fetchall():
            retrieved_chunks.append({
                "source": row[0], "content": row[1] or "",
                "metadata": row[2] if isinstance(row[2], dict) else {},
                "similarity": round(float(row[3]), 4),
                "table": "media_files",
            })

    if "structured_files" in existing_tables:
        cur.execute("""
            SELECT filepath, content, metadata,
                   1 - (summary_embedding <=> %s::vector) AS similarity
            FROM structured_files
            WHERE summary_embedding IS NOT NULL
            ORDER BY summary_embedding <=> %s::vector
            LIMIT 2
        """, (formatted_emb, formatted_emb))
        for row in cur.fetchall():
            retrieved_chunks.append({
                "source": row[0],
                "content": row[1][:2000] if row[1] else "",
                "metadata": row[2] if isinstance(row[2], dict) else {},
                "similarity": round(float(row[3]), 4),
                "table": "structured_files",
            })

    cur.close()
    conn.close()

    if not retrieved_chunks:
        return {
            "answer": "No relevant results found in the knowledge base.",
            "sources": [],
            "tables_searched": list(existing_tables),
        }

    retrieved_chunks.sort(key=lambda x: x["similarity"], reverse=True)

    # Build context for LLM
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Source {i}] (similarity: {chunk['similarity']}) -- {chunk['source']}\n"
            f"{chunk['content'][:1500]}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    rag_prompt = (
        "You are a helpful assistant answering questions based on retrieved documents.\n"
        "Use ONLY the context below to answer. If the context doesn't contain enough "
        "information, say so.\nCite which source(s) you used by referencing [Source N].\n\n"
        f"--- RETRIEVED CONTEXT ---\n{context_block}\n\n"
        f"--- USER QUESTION ---\n{question}\n\n"
        "Provide a clear, concise answer:"
    )

    try:
        response = llm.invoke(rag_prompt)
        answer = response.content
    except Exception as e:
        answer = f"LLM error: {str(e)}"

    sources = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        sources.append({
            "index": i,
            "source": chunk["source"],
            "similarity": chunk["similarity"],
            "table": chunk["table"],
            "preview": (chunk["content"][:200] + "...") if len(chunk["content"]) > 200 else chunk["content"],
        })

    return {
        "answer": answer,
        "sources": sources,
        "tables_searched": list(existing_tables),
    }
