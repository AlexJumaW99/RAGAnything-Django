# ingest_and_chat/views.py
"""
Django views that replicate the original FastAPI endpoints:
    GET  /            - Ingestion dashboard UI
    GET  /health/     - health check
    POST /ingest/     - run ingestion (SSE streaming response)
    POST /query/      - RAG query (JSON response)
    POST /stop/       - cancel running ingestion
"""

import os
import json
import logging

from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from . import services

logger = logging.getLogger("ingest_and_chat")


@require_GET
def dashboard(request):
    """Serve the ingestion dashboard UI."""
    return render(request, "ingest_and_chat/dashboard.html")


@require_GET
def health(request):
    """Health check endpoint."""
    return JsonResponse({"status": "ok", "pid": os.getpid()})


@csrf_exempt
@require_POST
def ingest(request):
    """
    Run the full RAG ingestion pipeline.
    Returns a streaming response (text/event-stream) with progress updates.
    Each line is a JSON object: {"event": "...", "data": {...}}
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    target_path = body.get("target_path", "").strip()
    if not target_path:
        return JsonResponse({"error": "target_path is required."}, status=400)

    if not os.path.exists(target_path):
        return JsonResponse(
            {"error": f"Path does not exist: {target_path}"}, status=400,
        )

    if services.is_ingestion_running():
        return JsonResponse(
            {"error": "An ingestion is already running."}, status=409,
        )

    response = StreamingHttpResponse(
        services.run_ingestion_stream(target_path),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@csrf_exempt
@require_POST
def query(request):
    """
    Run a RAG query: embed question, pgvector search, LLM answer.
    Returns JSON with the answer and source references.
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    question = body.get("question", "").strip()
    if not question:
        return JsonResponse({"error": "Question cannot be empty."}, status=400)

    try:
        result = services.query_rag(question)
        if result.get("error"):
            return JsonResponse({"error": result["error"]}, status=404)
        return JsonResponse(result)
    except Exception as e:
        logger.exception("Query failed")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def stop(request):
    """Cancel any running ingestion process."""
    was_running = services.cancel_ingestion()
    if was_running:
        return JsonResponse({
            "stopped": True,
            "message": "Cancel signal sent to ingestion.",
        })
    return JsonResponse({
        "stopped": False,
        "message": "No ingestion was running.",
    })
