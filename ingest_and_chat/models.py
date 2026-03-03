# ingest_and_chat/models.py
"""
This app manages its pgvector tables directly via psycopg2 (see nodes.setup_postgres).

The three RAG tables (document_chunks, media_files, structured_files) use the
pgvector extension for vector(384) columns, which is not natively supported by
the Django ORM. They are created/repaired automatically during ingestion.

If you later want Django ORM integration, consider using django-pgvector.
"""
