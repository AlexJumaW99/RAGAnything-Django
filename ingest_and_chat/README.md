# ingest_and_chat - Django App

A Django app that wraps a multi-agent RAG ingestion and query pipeline
powered by LangGraph, Gemini, pgvector, and sentence-transformers.

## Quick Setup

### 1. Install dependencies

Add the contents of `dependencies.txt` to your project requirements, then:

```bash
pip install -r requirements.txt
```

### 2. Add to INSTALLED_APPS

```python
# settings.py
INSTALLED_APPS = [
    ...
    "ingest_and_chat",
]
```

### 3. Include URLs

```python
# project/urls.py
from django.urls import path, include

urlpatterns = [
    ...
    path("api/rag/", include("ingest_and_chat.urls")),
]
```

### 4. Configure settings (optional)

All settings have env-var fallbacks so you can use a `.env` file instead.

```python
# settings.py
INGEST_AND_CHAT = {
    # PostgreSQL (pgvector database - separate from Django's DB)
    "PG_HOST": "localhost",
    "PG_PORT": 5432,
    "PG_USERNAME": "admin",
    "PG_PASSWORD": "your_password",
    "PG_DATABASE": "vector",

    # Google API key for Gemini LLM + multimodal
    "GOOGLE_API_KEY": "...",

    # Models
    "LLM_MODEL": "gemini-2.5-pro",
    "LLM_TEMPERATURE": 0.2,
    "EMBEDDING_MODEL": "all-MiniLM-L6-v2",

    # Processing
    "OCR_METHOD": "gemini",          # "gemini" or "tesseract"
    "TRANSCRIPTION_METHOD": "gemini", # "gemini" or "whisper"
    "MAX_BINARY_MB": 50,

    # Chunking
    "CHUNK_SIZE_TEXT": 1000,
    "CHUNK_OVERLAP_TEXT": 200,
    "CHUNK_SIZE_CODE": 1500,
    "CHUNK_OVERLAP_CODE": 200,

    # Query
    "SIMILARITY_THRESHOLD": 0.3,

    # Output paths (default to BASE_DIR/...)
    "PROCESSED_OUTPUT_ROOT": "/path/to/Processed_Output",
    "LOGS_ROOT": "/path/to/Logs",
}
```

Or set environment variables:
```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=admin
POSTGRES_PASSWORD=secret
POSTGRES_DATABASE=vector
GOOGLE_API_KEY=...
```

## API Endpoints

All endpoints are relative to the URL prefix you chose (e.g. `/api/rag/`).

| Method | Path       | Description                          |
|--------|------------|--------------------------------------|
| GET    | `health/`  | Health check                         |
| POST   | `ingest/`  | Run ingestion (SSE streaming)        |
| POST   | `query/`   | RAG query (JSON)                     |
| POST   | `stop/`    | Cancel running ingestion             |

### POST /ingest/

```json
{"target_path": "/path/to/files"}
```

Returns `text/event-stream` with JSON events:
- `status` - progress updates
- `log` - stdout lines from pipeline nodes
- `complete` - final results
- `error` - failure info
- `done` - stream finished

### POST /query/

```json
{"question": "What does the main.py file do?"}
```

Returns JSON:
```json
{
  "answer": "...",
  "sources": [...],
  "tables_searched": ["document_chunks", "media_files"]
}
```

## Architecture

```
ingest_and_chat/
    __init__.py
    apps.py              Django app config
    config.py            Settings + lazy model loaders
    states.py            LangGraph TypedDict state
    pipeline_graph.py    LangGraph StateGraph builder
    nodes.py             7 pipeline node functions
    services.py          Ingestion orchestration + RAG query
    views.py             Django views (health, ingest, query, stop)
    urls.py              URL routing
    utils.py             Tree builder, file stats, graph viz
    models.py            Empty (tables managed via psycopg2)
```

The LangGraph pipeline runs identically to the original standalone version.
The pgvector tables are created automatically during the first ingestion.

## Key changes from the standalone version

1. **Import fix**: The original code imported `llm`, `embedding_model`,
   `genai_client` as bare names from config, but config only exported
   getter functions. All nodes now correctly use `get_llm()`,
   `get_genai_client()`, `get_embedding_model()`.

2. **Django integration**: FastAPI replaced with Django views.
   `StreamingHttpResponse` replaces Starlette's `StreamingResponse`.

3. **Settings**: Config reads from `settings.INGEST_AND_CHAT` dict
   with env-var fallback, so it works with Django's config system.

4. **Logging**: `print()` statements in nodes replaced with
   `logging.getLogger("ingest_and_chat")` calls. The TeeWriter in
   services.py still captures stdout for SSE relay.

5. **No conda**: The conda environment auto-activation from `main.py`
   is removed; dependency management is handled by your Django project.
