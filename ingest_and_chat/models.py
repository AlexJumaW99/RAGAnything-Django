# ingest_and_chat/models.py
"""
This app manages its pgvector tables directly via psycopg2.

Database initialization is centralized in db.py, which handles:
  - Auto-creation of the PostgreSQL database if it doesn't exist
  - Auto-creation of the pgvector extension
  - Auto-creation of all tables (IF NOT EXISTS) with correct schema
  - Auto-creation of all indexes (B-tree, GIN, HNSW)
  - Thread-safe one-time initialization per process

Tables created and managed:

  ingestion_sessions  — Tracks each ingestion run (target path, status, timestamps)
  document_chunks     — Chunked text from code, docs, PDFs, DOCX, PPTX, configs
                        Includes vector(384) embedding + tsvector for hybrid search
  media_files         — Image OCR / audio transcriptions with embeddings
                        Binary files stored on disk (MEDIA_STORAGE_ROOT), not in DB
  structured_files    — CSV/XLSX schema descriptions + sample rows
                        Full files stored on disk for pandas code execution
  conversations       — Chat conversation metadata
  chat_messages       — Individual chat messages with role, content, sources

All data tables use the pgvector extension for vector(384) columns (all-MiniLM-L6-v2).
document_chunks, media_files, and structured_files have GIN indexes on
auto-generated tsvector columns for full-text search.
HNSW indexes are used for approximate nearest neighbor vector search.

Any code that needs a database connection should use:
    from .db import get_connection
    conn = get_connection()  # DB + tables guaranteed to exist

If you later want Django ORM integration, consider using django-pgvector.
"""
