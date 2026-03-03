# ingest_and_chat/views.py
"""
Django views:
    GET  /                  - Ingestion dashboard UI
    GET  /health/           - Health check
    POST /ingest/           - Run ingestion (SSE streaming response)
    POST /stop/             - Cancel running ingestion

    # Sessions
    GET  /sessions/         - List ingestion sessions
    GET  /sessions/<id>/    - Get session details
    DELETE /sessions/<id>/  - Delete session + all associated data

    # Chat
    POST /chat/             - Send a message (creates conversation if needed)
    GET  /conversations/                    - List conversations
    GET  /conversations/<id>/history/       - Get conversation messages
    DELETE /conversations/<id>/             - Delete conversation
"""

import os
import json
import logging

from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET

from . import services
from . import chat as chat_service

logger = logging.getLogger("ingest_and_chat")


# ---------------------------------------------------------------------------
# CORS helper
# ---------------------------------------------------------------------------

def _cors_headers(response):
    """Attach permissive CORS headers to any response."""
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def _options_response():
    """Return an empty 200 for preflight OPTIONS requests."""
    return _cors_headers(HttpResponse(status=200))


def _json_error(message, status=400):
    return _cors_headers(JsonResponse({"error": message}, status=status))


def _json_ok(data, status=200):
    return _cors_headers(JsonResponse(data, status=status))


def _parse_body(request):
    """Parse JSON body, returning (dict, None) or (None, error_response)."""
    try:
        body = json.loads(request.body)
        return body, None
    except (json.JSONDecodeError, ValueError):
        return None, _json_error("Invalid JSON body.")


# ---------------------------------------------------------------------------
# Dashboard & Health
# ---------------------------------------------------------------------------

@require_GET
def dashboard(request):
    """Serve the ingestion dashboard UI."""
    return render(request, "ingest_and_chat/dashboard.html")


@require_GET
def health(request):
    """Health check endpoint. Also verifies database readiness."""
    db_status = "ok"
    db_error = None
    try:
        from .db import ensure_db_ready
        ensure_db_ready()
    except Exception as e:
        db_status = "error"
        db_error = str(e)

    status = "ok" if db_status == "ok" else "degraded"
    return _json_ok({
        "status": status,
        "pid": os.getpid(),
        "database": db_status,
        "database_error": db_error,
    })


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

@csrf_exempt
def ingest(request):
    """Run the full RAG ingestion pipeline (SSE stream)."""
    if request.method == "OPTIONS":
        return _options_response()
    if request.method != "POST":
        return _json_error("Method not allowed.", 405)

    body, err = _parse_body(request)
    if err:
        return err

    target_path = body.get("target_path", "").strip()
    if not target_path:
        return _json_error("target_path is required.")
    if not os.path.exists(target_path):
        return _json_error(f"Path does not exist: {target_path}")
    if services.is_ingestion_running():
        return _json_error("An ingestion is already running.", 409)

    response = StreamingHttpResponse(
        services.run_ingestion_stream(target_path),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return _cors_headers(response)


@csrf_exempt
def stop(request):
    """Cancel any running ingestion process."""
    if request.method == "OPTIONS":
        return _options_response()
    if request.method != "POST":
        return _json_error("Method not allowed.", 405)

    was_running = services.cancel_ingestion()
    if was_running:
        return _json_ok({"stopped": True, "message": "Cancel signal sent to ingestion."})
    return _json_ok({"stopped": False, "message": "No ingestion was running."})


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@csrf_exempt
def sessions_list(request):
    """GET: list all ingestion sessions."""
    if request.method == "OPTIONS":
        return _options_response()
    if request.method != "GET":
        return _json_error("Method not allowed.", 405)

    try:
        result = chat_service.list_sessions()
        return _json_ok({"sessions": result}, status=200)
    except Exception as e:
        logger.exception("Failed to list sessions")
        return _json_error(str(e), 500)


@csrf_exempt
def session_detail(request, session_id):
    """GET: session details. DELETE: remove session and all data."""
    if request.method == "OPTIONS":
        return _options_response()

    if request.method == "GET":
        try:
            result = chat_service.get_session(session_id)
            if not result:
                return _json_error("Session not found.", 404)
            return _json_ok(result)
        except Exception as e:
            logger.exception("Failed to get session")
            return _json_error(str(e), 500)

    if request.method == "DELETE":
        try:
            deleted = chat_service.delete_session(session_id)
            if deleted:
                return _json_ok({"deleted": True, "session_id": session_id})
            return _json_error("Session not found.", 404)
        except Exception as e:
            logger.exception("Failed to delete session")
            return _json_error(str(e), 500)

    return _json_error("Method not allowed.", 405)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@csrf_exempt
def chat_send(request):
    """
    POST /chat/

    Body:
        {
            "question": "...",
            "conversation_id": "uuid" (optional — created if omitted),
            "session_id": "uuid"      (optional — scope search to session)
        }

    Returns:
        {
            "conversation_id": "...",
            "answer": "...",
            "sources": [...],
            "tool_results": [...],
            "tables_searched": [...]
        }
    """
    if request.method == "OPTIONS":
        return _options_response()
    if request.method != "POST":
        return _json_error("Method not allowed.", 405)

    body, err = _parse_body(request)
    if err:
        return err

    question = body.get("question", "").strip()
    if not question:
        return _json_error("question is required.")

    conversation_id = body.get("conversation_id")
    session_id = body.get("session_id")

    try:
        # Create conversation if not provided
        if not conversation_id:
            conv = chat_service.create_conversation(
                session_id=session_id,
                title=question[:100],
            )
            conversation_id = str(conv["id"])

        result = chat_service.chat(
            conversation_id=conversation_id,
            question=question,
            session_id=session_id,
        )

        if result.get("error"):
            return _json_error(result["error"], 404)

        return _json_ok({
            "conversation_id": conversation_id,
            **result,
        })

    except Exception as e:
        logger.exception("Chat failed")
        return _json_error(str(e), 500)


@csrf_exempt
def conversations_list(request):
    """GET: list conversations, optionally filtered by ?session_id=..."""
    if request.method == "OPTIONS":
        return _options_response()
    if request.method != "GET":
        return _json_error("Method not allowed.", 405)

    try:
        session_id = request.GET.get("session_id")
        result = chat_service.list_conversations(session_id=session_id)
        return _json_ok({"conversations": result})
    except Exception as e:
        logger.exception("Failed to list conversations")
        return _json_error(str(e), 500)


@csrf_exempt
def conversation_history(request, conversation_id):
    """GET: get all messages in a conversation."""
    if request.method == "OPTIONS":
        return _options_response()
    if request.method != "GET":
        return _json_error("Method not allowed.", 405)

    try:
        limit = request.GET.get("limit")
        limit = int(limit) if limit else None
        messages = chat_service.get_conversation_history(conversation_id, limit=limit)
        return _json_ok({"conversation_id": conversation_id, "messages": messages})
    except Exception as e:
        logger.exception("Failed to get conversation history")
        return _json_error(str(e), 500)


@csrf_exempt
def conversation_delete(request, conversation_id):
    """DELETE: remove a conversation and its messages."""
    if request.method == "OPTIONS":
        return _options_response()
    if request.method != "DELETE":
        return _json_error("Method not allowed.", 405)

    try:
        deleted = chat_service.delete_conversation(conversation_id)
        if deleted:
            return _json_ok({"deleted": True, "conversation_id": conversation_id})
        return _json_error("Conversation not found.", 404)
    except Exception as e:
        logger.exception("Failed to delete conversation")
        return _json_error(str(e), 500)
