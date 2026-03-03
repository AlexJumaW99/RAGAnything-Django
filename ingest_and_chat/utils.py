# ingest_and_chat/utils.py
"""
Utility helpers: project tree builder, file stats, graph visualisation.
"""

import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger("ingest_and_chat")

# ---------------------------------------------------------------------------
# Graph Visualization
# ---------------------------------------------------------------------------

def save_graph_image(app, thread_id="default"):
    """Saves a Mermaid PNG of the graph structure."""
    from .config import LOGS_ROOT

    output_dir = os.path.join(LOGS_ROOT, "Graphs")
    os.makedirs(output_dir, exist_ok=True)

    try:
        png_data = app.get_graph().draw_mermaid_png()
        output_path = os.path.join(output_dir, f"graph_{thread_id}.png")
        with open(output_path, "wb") as f:
            f.write(png_data)
        logger.info("Graph saved to: %s", output_path)
    except Exception as e:
        logger.warning("Could not generate graph image: %s", e)


# ---------------------------------------------------------------------------
# Project / Directory Tree Builder
# ---------------------------------------------------------------------------

_TREE_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", "dist", "build", ".egg-info",
}


def build_project_tree(
    root_path: str,
    prefix: str = "",
    max_depth: int = 4,
    _depth: int = 0,
) -> str:
    """Builds a pretty-printed directory tree string."""
    if _depth > max_depth:
        return ""

    if os.path.isfile(root_path):
        return os.path.basename(root_path) + "\n"

    entries = sorted(os.listdir(root_path))
    entries = [e for e in entries if not e.startswith(".") and e not in _TREE_SKIP_DIRS]

    lines = []
    if _depth == 0:
        lines.append(os.path.basename(os.path.abspath(root_path)) + "/")

    for i, entry in enumerate(entries):
        full_path = os.path.join(root_path, entry)
        is_last = i == len(entries) - 1
        connector = "+-- " if is_last else "|-- "
        extension = "    " if is_last else "|   "

        if os.path.isdir(full_path):
            lines.append(f"{prefix}{connector}{entry}/")
            subtree = build_project_tree(
                full_path,
                prefix=prefix + extension,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            if subtree:
                lines.append(subtree.rstrip("\n"))
        else:
            lines.append(f"{prefix}{connector}{entry}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File Stat Helpers
# ---------------------------------------------------------------------------

def _epoch_to_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def get_file_stats(filepath: str) -> dict:
    """Returns creation date, modified date, and size for a file."""
    try:
        stat = os.stat(filepath)
        return {
            "file_size_bytes": stat.st_size,
            "file_created_date": _epoch_to_iso(
                getattr(stat, "st_birthtime", stat.st_ctime)
            ),
            "file_modified_date": _epoch_to_iso(stat.st_mtime),
        }
    except Exception:
        return {
            "file_size_bytes": 0,
            "file_created_date": None,
            "file_modified_date": None,
        }
