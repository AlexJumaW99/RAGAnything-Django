# ingest_and_chat/states.py
import threading
from typing import Any, Dict, List, TypedDict, Literal, Optional


class FileMetadata(TypedDict):
    """Structured output format expected from Gemini for file metadata."""
    summary: str
    topics: List[str]
    key_entities: List[str]
    content_category: Literal[
        "code", "documentation", "data", "media",
        "configuration", "other", "unknown",
    ]
    quality_notes: str


class RAGIngestionState(TypedDict):
    """Full state for the multi-agent RAG ingestion pipeline."""

    # Session and Input
    session_id: str
    target_path: str
    project_tree: Optional[str]
    output_dir: str

    # File Classification
    classified_files: Optional[Dict[str, List[Dict[str, Any]]]]

    # Processed Outputs
    processed_documents: Optional[List[Dict[str, Any]]]
    processed_media: Optional[List[Dict[str, Any]]]
    processed_structured: Optional[List[Dict[str, Any]]]

    # Metadata (LLM-enriched)
    file_metadata: Optional[Dict[str, Dict[str, Any]]]

    # PostgreSQL Configuration
    pg_host: str
    pg_port: int
    pg_username: str
    pg_password: str
    pg_database: str

    # Execution Tracking
    records_inserted: int
    current_step: Optional[str]
    steps_completed: Optional[List[str]]

    # Error Handling
    has_error: bool
    errors: Optional[List[str]]
    error_log_path: Optional[str]
    debug_summary: Optional[str]

    # Command Outputs
    last_command: Optional[str]
    last_stdout: Optional[str]
    last_stderr: Optional[str]


# Global cancellation event shared between views and pipeline nodes.
cancellation_event = threading.Event()
