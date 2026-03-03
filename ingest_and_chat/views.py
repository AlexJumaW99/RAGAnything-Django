# ingest_and_chat/views.py
"""
Django views:
    GET  /            - Ingestion dashboard UI
    GET  /health/     - health check
    POST /ingest/     - run ingestion (SSE streaming response)
    POST /query/      - RAG query (JSON response)
    POST /stop/       - cancel running ingestion
"""

import os
import json
import logging

from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET

from . import services

logger = logging.getLogger("ingest_and_chat")


# ---------------------------------------------------------------------------
# CORS helper — lets the browser talk to the API even when the origin
# differs slightly (e.g. 127.0.0.1 vs localhost).
# ---------------------------------------------------------------------------

def _cors_headers(response):
    """Attach permissive CORS headers to any response."""
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def _options_response():
    """Return an empty 200 for preflight OPTIONS requests."""
    return _cors_headers(HttpResponse(status=200))


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

@require_GET
def dashboard(request):
    """Serve the ingestion dashboard UI."""
    return render(request, "ingest_and_chat/dashboard.html")


@require_GET
def health(request):
    """Health check endpoint."""
    resp = JsonResponse({"status": "ok", "pid": os.getpid()})
    return _cors_headers(resp)


@csrf_exempt
def ingest(request):
    """
    Run the full RAG ingestion pipeline.
    Accepts POST (returns SSE stream) and OPTIONS (CORS preflight).
    """
    if request.method == "OPTIONS":
        return _options_response()

    if request.method != "POST":
        return _cors_headers(JsonResponse({"error": "Method not allowed."}, status=405))

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return _cors_headers(JsonResponse({"error": "Invalid JSON body."}, status=400))

    target_path = body.get("target_path", "").strip()
    if not target_path:
        return _cors_headers(JsonResponse({"error": "target_path is required."}, status=400))

    if not os.path.exists(target_path):
        return _cors_headers(
            JsonResponse({"error": f"Path does not exist: {target_path}"}, status=400)
        )

    if services.is_ingestion_running():
        return _cors_headers(
            JsonResponse({"error": "An ingestion is already running."}, status=409)
        )

    response = StreamingHttpResponse(
        services.run_ingestion_stream(target_path),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return _cors_headers(response)


@csrf_exempt
def query(request):
    """
    Run a RAG query: embed question, pgvector search, LLM answer.
    """
    if request.method == "OPTIONS":
        return _options_response()

    if request.method != "POST":
        return _cors_headers(JsonResponse({"error": "Method not allowed."}, status=405))

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return _cors_headers(JsonResponse({"error": "Invalid JSON body."}, status=400))

    question = body.get("question", "").strip()
    if not question:
        return _cors_headers(JsonResponse({"error": "Question cannot be empty."}, status=400))

    try:
        result = services.query_rag(question)
        if result.get("error"):
            return _cors_headers(JsonResponse({"error": result["error"]}, status=404))
        return _cors_headers(JsonResponse(result))
    except Exception as e:
        logger.exception("Query failed")
        return _cors_headers(JsonResponse({"error": str(e)}, status=500))


@csrf_exempt
def stop(request):
    """Cancel any running ingestion process."""
    if request.method == "OPTIONS":
        return _options_response()

    if request.method != "POST":
        return _cors_headers(JsonResponse({"error": "Method not allowed."}, status=405))

    was_running = services.cancel_ingestion()
    if was_running:
        return _cors_headers(JsonResponse({
            "stopped": True,
            "message": "Cancel signal sent to ingestion.",
        }))
    return _cors_headers(JsonResponse({
        "stopped": False,
        "message": "No ingestion was running.",
    }))
