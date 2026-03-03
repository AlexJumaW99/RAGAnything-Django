# ingest_and_chat - Django App

A Django app that wraps a multi-agent RAG ingestion and query pipeline
powered by LangGraph, Gemini, pgvector, and sentence-transformers.

## What's New (v2)

- **Ingestion sessions** — every ingestion creates a tracked session in the DB. Query, list, or delete sessions and all their associated data.
- **Hybrid search** — vector cosine similarity (pgvector HNSW) combined with full-text search (GIN index on tsvector). Configurable weights.
- **Chat with memory** — conversations track multi-turn history. The LLM sees prior messages when answering follow-ups.
- **Pandas code execution** — for CSV/XLSX files, the LLM generates and executes pandas code in a sandboxed environment to answer analytical questions.
- **Disk-based media storage** — images and audio are stored on disk (not BYTEA in Postgres), reducing DB bloat.
- **Structured data schema** — CSV/XLSX files store schema descriptions + sample rows instead of full content dumps. Original files on disk for pandas access.
- **HNSW vector indexes** — approximate nearest neighbor search instead of sequential scan.
- **Full-text GIN indexes** — exact keyword matching for function names, error codes, etc.
- **Session-scoped queries** — optionally restrict search to a specific ingestion session.

## Quick Setup

### 1. Install dependencies

```bash
pip install -r ingest_and_chat/dependencies.txt
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
    "OCR_METHOD": "gemini",
    "TRANSCRIPTION_METHOD": "gemini",
    "MAX_BINARY_MB": 50,

    # Chunking
    "CHUNK_SIZE_TEXT": 1000,
    "CHUNK_OVERLAP_TEXT": 200,
    "CHUNK_SIZE_CODE": 1500,
    "CHUNK_OVERLAP_CODE": 200,

    # Query
    "SIMILARITY_THRESHOLD": 0.3,
    "HYBRID_VECTOR_WEIGHT": 0.7,      # Weight for vector similarity
    "HYBRID_TEXT_WEIGHT": 0.3,         # Weight for full-text search

    # Chat
    "CHAT_HISTORY_LIMIT": 20,          # Messages to include as context
    "PANDAS_EXEC_TIMEOUT": 30,         # Seconds before killing pandas execution

    # Storage paths
    "PROCESSED_OUTPUT_ROOT": "/path/to/Processed_Output",
    "LOGS_ROOT": "/path/to/Logs",
    "MEDIA_STORAGE_ROOT": "/path/to/Media_Storage",
}
```

### 5. PostgreSQL requirements

Your PostgreSQL instance must have the `pgvector` extension installed:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Tables are created automatically during the first ingestion. **This version requires a fresh database** — the schema has changed from v1. If you have existing data, back it up and drop the old tables before running.

## API Endpoints

All endpoints are relative to the URL prefix (e.g. `/api/rag/`).

### Pipeline

| Method | Path       | Description                          |
|--------|------------|--------------------------------------|
| GET    | `health/`  | Health check                         |
| POST   | `ingest/`  | Run ingestion (SSE streaming)        |
| POST   | `stop/`    | Cancel running ingestion             |

### Sessions

| Method | Path                   | Description                          |
|--------|------------------------|--------------------------------------|
| GET    | `sessions/`            | List all ingestion sessions          |
| GET    | `sessions/<uuid>/`     | Get session details                  |
| DELETE | `sessions/<uuid>/`     | Delete session + all associated data |

### Chat

| Method | Path                                    | Description                    |
|--------|-----------------------------------------|--------------------------------|
| POST   | `chat/`                                 | Send message, get RAG response |
| GET    | `conversations/`                        | List conversations             |
| GET    | `conversations/<uuid>/history/`         | Get conversation messages      |
| DELETE | `conversations/<uuid>/`                 | Delete conversation            |

### POST /ingest/

```json
{"target_path": "/path/to/files"}
```

Returns `text/event-stream` with JSON events. The `complete` event now includes
`db_session_id` which you can use to scope future chat queries.

### POST /chat/

```json
{
    "question": "What does the main.py file do?",
    "conversation_id": "optional-uuid",
    "session_id": "optional-uuid-to-scope-search"
}
```

If `conversation_id` is omitted, a new conversation is created automatically.
If `session_id` is provided, search is restricted to data from that ingestion.

Returns:
```json
{
    "conversation_id": "...",
    "answer": "...",
    "sources": [
        {
            "index": 1,
            "source": "/path/to/file.py",
            "score": 0.87,
            "table": "document_chunks",
            "preview": "first 200 chars..."
        }
    ],
    "tool_results": [
        {
            "file": "data.csv",
            "success": true,
            "result": "Average revenue: $1.2M",
            "code": "import pandas as pd\n..."
        }
    ],
    "tables_searched": ["document_chunks", "media_files", "structured_files"]
}
```

## Database Schema

```
ingestion_sessions     — Tracks each ingestion run
  ├── document_chunks  — Text chunks with vector + tsvector (CASCADE delete)
  ├── media_files      — Transcripts + disk path (CASCADE delete)
  └── structured_files — Schema + samples + disk path (CASCADE delete)

conversations          — Chat conversation metadata
  └── chat_messages    — Messages with role, content, sources, tool_calls
```

### Key design decisions

- **Hybrid search**: Every content table has both a `vector(384)` column with HNSW index AND a `tsvector` column with GIN index. Queries combine both signals with configurable weights.
- **No BYTEA**: Media files are stored on disk under `MEDIA_STORAGE_ROOT/<session_id>/`. The DB only stores the path.
- **Schema-first structured data**: Instead of dumping full CSV content, we store a schema description, data types, and 5 sample rows. The original file is on disk for pandas execution.
- **Session scoping**: All data tables have a `session_id` FK. Chat queries can optionally restrict search to one session.

## Architecture

```
ingest_and_chat/
    __init__.py
    apps.py              Django app config
    config.py            Settings + lazy model loaders
    states.py            LangGraph TypedDict state
    pipeline_graph.py    LangGraph StateGraph builder
    nodes.py             7 pipeline node functions (new schema)
    services.py          Ingestion orchestration (SSE streaming)
    chat.py              Chat engine, hybrid search, pandas execution
    views.py             Django views (pipeline + chat + sessions)
    urls.py              URL routing
    utils.py             Tree builder, file stats, graph viz
    models.py            Schema documentation (tables via psycopg2)
```
