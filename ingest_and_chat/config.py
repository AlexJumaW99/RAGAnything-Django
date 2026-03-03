# ingest_and_chat/config.py
"""
Configuration for the RAG pipeline.

Reads from Django settings (INGEST_AND_CHAT dict) when available,
falling back to environment variables / sensible defaults.
All heavy models are lazy-loaded on first access.
"""

import os
import logging

from django.conf import settings

logger = logging.getLogger("ingest_and_chat")

# ---------------------------------------------------------------------------
# Helper: read a setting from Django settings dict, then env, then default
# ---------------------------------------------------------------------------

_DJANGO_CONF: dict = getattr(settings, "INGEST_AND_CHAT", {})


def _setting(key: str, env_key: str | None = None, default=None):
    """
    Resolution order:
      1. settings.INGEST_AND_CHAT[key]
      2. os.environ[env_key]
      3. default
    """
    val = _DJANGO_CONF.get(key)
    if val is not None:
        return val
    if env_key:
        val = os.getenv(env_key)
        if val is not None:
            return val
    return default


# ---------------------------------------------------------------------------
# PostgreSQL Constants
# ---------------------------------------------------------------------------
PG_HOST = _setting("PG_HOST", "POSTGRES_HOST", "localhost")
PG_PORT = int(_setting("PG_PORT", "POSTGRES_PORT", 5432))
PG_USERNAME = _setting("PG_USERNAME", "POSTGRES_USER", "admin")
PG_PASSWORD = _setting("PG_PASSWORD", "POSTGRES_PASSWORD", "")
PG_DATABASE = _setting("PG_DATABASE", "POSTGRES_DATABASE", "vector")

if not PG_PASSWORD:
    logger.warning("POSTGRES_PASSWORD not configured — database operations may fail.")

# ---------------------------------------------------------------------------
# Media Processing Configuration
# ---------------------------------------------------------------------------
OCR_METHOD = str(_setting("OCR_METHOD", "OCR_METHOD", "gemini")).lower()
TRANSCRIPTION_METHOD = str(_setting("TRANSCRIPTION_METHOD", "TRANSCRIPTION_METHOD", "gemini")).lower()
MAX_BINARY_MB = int(_setting("MAX_BINARY_MB", "MAX_BINARY_MB", 50))

# Chunking
CHUNK_SIZE_TEXT = int(_setting("CHUNK_SIZE_TEXT", "CHUNK_SIZE_TEXT", 1000))
CHUNK_OVERLAP_TEXT = int(_setting("CHUNK_OVERLAP_TEXT", "CHUNK_OVERLAP_TEXT", 200))
CHUNK_SIZE_CODE = int(_setting("CHUNK_SIZE_CODE", "CHUNK_SIZE_CODE", 1500))
CHUNK_OVERLAP_CODE = int(_setting("CHUNK_OVERLAP_CODE", "CHUNK_OVERLAP_CODE", 200))

# Structured data row cap
STRUCTURED_MAX_ROWS = int(_setting("STRUCTURED_MAX_ROWS", "STRUCTURED_MAX_ROWS", 500))

# Query similarity threshold
SIMILARITY_THRESHOLD = float(_setting("SIMILARITY_THRESHOLD", "SIMILARITY_THRESHOLD", 0.3))

# ---------------------------------------------------------------------------
# Processed output root — defaults to <project_root>/Processed_Output
# ---------------------------------------------------------------------------
PROCESSED_OUTPUT_ROOT = _setting(
    "PROCESSED_OUTPUT_ROOT",
    "PROCESSED_OUTPUT_ROOT",
    os.path.join(settings.BASE_DIR, "Processed_Output"),
)

LOGS_ROOT = _setting(
    "LOGS_ROOT",
    "LOGS_ROOT",
    os.path.join(settings.BASE_DIR, "Logs"),
)

# ---------------------------------------------------------------------------
# File Type Classifications
# ---------------------------------------------------------------------------
TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".html", ".htm", ".log", ".tex"}
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh", ".bash",
    ".r", ".sql", ".css", ".scss", ".less", ".vue", ".svelte",
}
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".xml", ".env", ".ini", ".cfg", ".conf"}
PDF_EXTENSIONS = {".pdf"}
OFFICE_EXTENSIONS = {".docx", ".pptx"}
STRUCTURED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}

CODE_LANGUAGE_MAP = {
    ".py": "python", ".js": "js", ".ts": "ts", ".jsx": "js", ".tsx": "ts",
    ".java": "java", ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
    ".sh": "python", ".bash": "python",
    ".sql": "python", ".r": "python",
    ".html": "html", ".htm": "html", ".css": "python",
    ".scss": "python", ".less": "python",
    ".vue": "html", ".svelte": "html",
}

MIME_TYPE_MAP = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
    ".ogg": "audio/ogg", ".flac": "audio/flac", ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
}

# ---------------------------------------------------------------------------
# LLM & Embedding Model — Lazy Loading
# ---------------------------------------------------------------------------

_llm = None
_genai_client = None
_embedding_model = None
_models_loaded = {"llm": False, "genai": False, "embedding": False}


def _connect_to_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=_setting("LLM_MODEL", "LLM_MODEL", "gemini-2.5-pro"),
        temperature=float(_setting("LLM_TEMPERATURE", "LLM_TEMPERATURE", 0.2)),
        timeout=120,
        max_retries=4,
    )


def _connect_genai_client():
    from google import genai
    api_key = _setting("GOOGLE_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not configured — multimodal features will fail.")
        return None
    return genai.Client(api_key=api_key)


def _load_embedding_model():
    from sentence_transformers import SentenceTransformer
    model_name = _setting("EMBEDDING_MODEL", "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def get_llm():
    """Lazy accessor for the LLM singleton."""
    global _llm
    if not _models_loaded["llm"]:
        logger.info("Loading LLM...")
        _llm = _connect_to_llm()
        _models_loaded["llm"] = True
    return _llm


def get_genai_client():
    """Lazy accessor for the genai client singleton."""
    global _genai_client
    if not _models_loaded["genai"]:
        logger.info("Initializing Gemini genai client...")
        _genai_client = _connect_genai_client()
        _models_loaded["genai"] = True
    return _genai_client


def get_embedding_model():
    """Lazy accessor for the embedding model singleton."""
    global _embedding_model
    if not _models_loaded["embedding"]:
        logger.info("Loading embedding model...")
        _embedding_model = _load_embedding_model()
        _models_loaded["embedding"] = True
    return _embedding_model
