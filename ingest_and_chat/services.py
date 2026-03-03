# ingest_and_chat/services.py
"""
Business logic layer: ingestion orchestration.

The RAG query / chat engine has been moved to chat.py.
This module handles only the ingestion pipeline execution.
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
                "db_session_id": None,
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
                def isatty(self):
                    return False
                def __getattr__(self, name):
                    return getattr(self.original, name)

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
                    "db_session_id": final_state.get("db_session_id"),
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
        except Exception:
            yield f"data: {json.dumps({'event': 'keepalive', 'data': {}})}\n"
