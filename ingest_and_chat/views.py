# ingest_and_chat/views.py
"""
Django views:
    GET  /                  - Ingestion dashboard UI
    GET  /health/           - Health check
    POST /ingest/           - Run ingestion (SSE streaming response)
                              Accepts JSON {"target_path": "..."} OR
                              multipart file uploads
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
import uuid
import shutil
import logging
import tempfile

from django.conf import settings
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET

from . import services
from . import chat as chat_service

logger = logging.getLogger("ingest_and_chat")

# ---------------------------------------------------------------------------
# Upload directory (for browser file uploads)
# ---------------------------------------------------------------------------
UPLOAD_ROOT = getattr(settings, "INGEST_AND_CHAT", {}).get(
    "UPLOAD_ROOT",
    os.environ.get("UPLOAD_ROOT", os.path.join(settings.BASE_DIR, "Uploads")),
)


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
# File Upload Helper
# ---------------------------------------------------------------------------

def _save_uploaded_files(request) -> tuple:
    """
    Save uploaded files from a multipart request to a temporary directory,
    preserving directory structure when relative paths are provided.

    The frontend sends two parallel form arrays:
      - "files": the actual file blobs
      - "relative_paths": the relative path for each file (e.g. "project/src/main.py")

    When relative_paths are present, files are placed in their original
    directory structure. Otherwise they're saved flat.

    Returns (upload_dir_path, error_message).
    If successful, error_message is None.
    """
    files = request.FILES.getlist("files")
    if not files:
        return None, "No files were uploaded."

    # Get relative paths (one per file, in matching order)
    relative_paths = request.POST.getlist("relative_paths")

    # Create a unique upload directory
    upload_id = uuid.uuid4().hex[:12]
    upload_dir = os.path.join(UPLOAD_ROOT, f"upload_{upload_id}")
    os.makedirs(upload_dir, exist_ok=True)

    try:
        for i, uploaded_file in enumerate(files):
            # Determine the destination path
            if i < len(relative_paths) and relative_paths[i]:
                # Use the relative path from the frontend
                rel_path = relative_paths[i]
                # Security: sanitize to prevent directory traversal
                # Remove any leading slashes or .. components
                parts = rel_path.replace("\\", "/").split("/")
                safe_parts = [
                    p for p in parts
                    if p and p != ".." and p != "." and not p.startswith("~")
                ]
                if not safe_parts:
                    safe_parts = [uploaded_file.name or f"unnamed_{uuid.uuid4().hex[:8]}"]
                rel_path = os.path.join(*safe_parts)
            else:
                # Flat file — just use the filename
                filename = os.path.basename(uploaded_file.name or "")
                if not filename:
                    filename = f"unnamed_{uuid.uuid4().hex[:8]}"
                rel_path = filename

            dest_path = os.path.join(upload_dir, rel_path)

            # Create parent directories
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Handle name collisions
            if os.path.exists(dest_path):
                name, ext = os.path.splitext(dest_path)
                dest_path = f"{name}_{uuid.uuid4().hex[:6]}{ext}"

            with open(dest_path, "wb") as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            logger.info("Saved uploaded file: %s -> %s", rel_path, dest_path)

        return upload_dir, None

    except Exception as e:
        # Clean up on failure
        try:
            shutil.rmtree(upload_dir, ignore_errors=True)
        except Exception:
            pass
        return None, f"Failed to save uploaded files: {str(e)}"


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
    """
    Run the full RAG ingestion pipeline (SSE stream).

    Accepts EITHER:
      - JSON body: {"target_path": "/absolute/path/to/files"}
      - Multipart form: files uploaded via browser file picker
    """
    if request.method == "OPTIONS":
        return _options_response()
    if request.method != "POST":
        return _json_error("Method not allowed.", 405)

    if services.is_ingestion_running():
        return _json_error("An ingestion is already running.", 409)

    target_path = None
    is_upload = False

    # Check if this is a multipart file upload
    if request.content_type and "multipart" in request.content_type:
        upload_dir, upload_err = _save_uploaded_files(request)
        if upload_err:
            return _json_error(upload_err)
        target_path = upload_dir
        is_upload = True
        logger.info("File upload ingestion: %d file(s) saved to %s",
                     len(request.FILES.getlist("files")), upload_dir)
    else:
        # JSON body with target_path
        body, err = _parse_body(request)
        if err:
            return err
        target_path = body.get("target_path", "").strip()
        if not target_path:
            return _json_error("target_path is required.")
        if not os.path.exists(target_path):
            return _json_error(f"Path does not exist: {target_path}")

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
