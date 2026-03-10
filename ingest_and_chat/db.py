# ingest_and_chat/db.py
"""
Centralized database initialization and connection management.

Provides:
  - Auto-creation of the PostgreSQL database if it doesn't exist
  - Auto-creation of pgvector extension
  - Auto-creation of all tables with correct schema (IF NOT EXISTS)
  - Auto-migration of missing columns on existing tables
  - Auto-creation of all indexes (B-tree, GIN, HNSW)
  - Thread-safe initialization that runs once per process
  - A get_connection() helper that guarantees the DB is ready

Usage:
    from .db import get_connection

    conn = get_connection()  # DB + tables guaranteed to exist
    cur = conn.cursor()
    ...
"""

import threading
import logging

import psycopg2
from psycopg2 import sql as psycopg2_sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from .config import PG_HOST, PG_PORT, PG_USERNAME, PG_PASSWORD, PG_DATABASE

logger = logging.getLogger("ingest_and_chat")

# ---------------------------------------------------------------------------
# Thread-safe initialization state
# ---------------------------------------------------------------------------

_init_lock = threading.Lock()
_initialized = False


# ===========================================================================
# Table Definitions (single source of truth for the entire app)
# ===========================================================================

# Order matters: parent tables before children (FK references).
TABLE_CREATE_ORDER = [
    "ingestion_sessions",
    "document_chunks",
    "media_files",
    "structured_files",
    "conversations",
    "chat_messages",
]

TABLE_DEFINITIONS = {
    "ingestion_sessions": """
        CREATE TABLE IF NOT EXISTS ingestion_sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT,
            target_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running',
            file_count INTEGER DEFAULT 0,
            records_inserted INTEGER DEFAULT 0,
            config JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        );
    """,

    "document_chunks": """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES ingestion_sessions(id) ON DELETE CASCADE,
            filepath TEXT NOT NULL,
            file_type TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            total_chunks INTEGER NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB,
            embedding vector(384),
            search_vector tsvector GENERATED ALWAYS AS (
                to_tsvector('english', content)
            ) STORED,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """,

    "media_files": """
        CREATE TABLE IF NOT EXISTS media_files (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES ingestion_sessions(id) ON DELETE CASCADE,
            filepath TEXT NOT NULL,
            file_type TEXT NOT NULL,
            storage_path TEXT,
            transcript TEXT,
            metadata JSONB,
            transcript_embedding vector(384),
            search_vector tsvector GENERATED ALWAYS AS (
                to_tsvector('english', COALESCE(transcript, ''))
            ) STORED,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """,

    "structured_files": """
        CREATE TABLE IF NOT EXISTS structured_files (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES ingestion_sessions(id) ON DELETE CASCADE,
            filepath TEXT NOT NULL,
            file_type TEXT NOT NULL,
            storage_path TEXT,
            schema_description TEXT,
            sample_rows JSONB,
            column_names JSONB,
            dtypes JSONB,
            row_count INTEGER,
            metadata JSONB,
            summary_embedding vector(384),
            search_vector tsvector GENERATED ALWAYS AS (
                to_tsvector('english', COALESCE(schema_description, ''))
            ) STORED,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """,

    "conversations": """
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES ingestion_sessions(id) ON DELETE SET NULL,
            title TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """,

    "chat_messages": """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id UUID NOT NULL,
            session_id UUID REFERENCES ingestion_sessions(id) ON DELETE SET NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources JSONB,
            tool_calls JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """,
}


# ===========================================================================
# Expected Column Definitions (for migration of existing tables)
# ===========================================================================
# Maps table_name -> list of (column_name, column_type_sql, is_generated)
# This is used to detect and add missing columns on existing tables.
# Generated columns (like search_vector) need special handling.

EXPECTED_COLUMNS = {
    "ingestion_sessions": [
        ("id", "UUID PRIMARY KEY DEFAULT gen_random_uuid()", False),
        ("name", "TEXT", False),
        ("target_path", "TEXT NOT NULL DEFAULT ''", False),
        ("status", "TEXT NOT NULL DEFAULT 'running'", False),
        ("file_count", "INTEGER DEFAULT 0", False),
        ("records_inserted", "INTEGER DEFAULT 0", False),
        ("config", "JSONB", False),
        ("created_at", "TIMESTAMPTZ DEFAULT NOW()", False),
        ("completed_at", "TIMESTAMPTZ", False),
    ],
    "document_chunks": [
        ("id", "UUID PRIMARY KEY DEFAULT gen_random_uuid()", False),
        ("session_id", "UUID REFERENCES ingestion_sessions(id) ON DELETE CASCADE", False),
        ("filepath", "TEXT NOT NULL DEFAULT ''", False),
        ("file_type", "TEXT NOT NULL DEFAULT ''", False),
        ("chunk_index", "INTEGER NOT NULL DEFAULT 0", False),
        ("total_chunks", "INTEGER NOT NULL DEFAULT 0", False),
        ("content", "TEXT NOT NULL DEFAULT ''", False),
        ("metadata", "JSONB", False),
        ("embedding", "vector(384)", False),
        ("search_vector", "tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED", True),
        ("created_at", "TIMESTAMPTZ DEFAULT NOW()", False),
    ],
    "media_files": [
        ("id", "UUID PRIMARY KEY DEFAULT gen_random_uuid()", False),
        ("session_id", "UUID REFERENCES ingestion_sessions(id) ON DELETE CASCADE", False),
        ("filepath", "TEXT NOT NULL DEFAULT ''", False),
        ("file_type", "TEXT NOT NULL DEFAULT ''", False),
        ("storage_path", "TEXT", False),
        ("transcript", "TEXT", False),
        ("metadata", "JSONB", False),
        ("transcript_embedding", "vector(384)", False),
        ("search_vector", "tsvector GENERATED ALWAYS AS (to_tsvector('english', COALESCE(transcript, ''))) STORED", True),
        ("created_at", "TIMESTAMPTZ DEFAULT NOW()", False),
    ],
    "structured_files": [
        ("id", "UUID PRIMARY KEY DEFAULT gen_random_uuid()", False),
        ("session_id", "UUID REFERENCES ingestion_sessions(id) ON DELETE CASCADE", False),
        ("filepath", "TEXT NOT NULL DEFAULT ''", False),
        ("file_type", "TEXT NOT NULL DEFAULT ''", False),
        ("storage_path", "TEXT", False),
        ("schema_description", "TEXT", False),
        ("sample_rows", "JSONB", False),
        ("column_names", "JSONB", False),
        ("dtypes", "JSONB", False),
        ("row_count", "INTEGER", False),
        ("metadata", "JSONB", False),
        ("summary_embedding", "vector(384)", False),
        ("search_vector", "tsvector GENERATED ALWAYS AS (to_tsvector('english', COALESCE(schema_description, ''))) STORED", True),
        ("created_at", "TIMESTAMPTZ DEFAULT NOW()", False),
    ],
    "conversations": [
        ("id", "UUID PRIMARY KEY DEFAULT gen_random_uuid()", False),
        ("session_id", "UUID REFERENCES ingestion_sessions(id) ON DELETE SET NULL", False),
        ("title", "TEXT", False),
        ("created_at", "TIMESTAMPTZ DEFAULT NOW()", False),
        ("updated_at", "TIMESTAMPTZ DEFAULT NOW()", False),
    ],
    "chat_messages": [
        ("id", "UUID PRIMARY KEY DEFAULT gen_random_uuid()", False),
        ("conversation_id", "UUID NOT NULL DEFAULT gen_random_uuid()", False),
        ("session_id", "UUID REFERENCES ingestion_sessions(id) ON DELETE SET NULL", False),
        ("role", "TEXT NOT NULL DEFAULT ''", False),
        ("content", "TEXT NOT NULL DEFAULT ''", False),
        ("sources", "JSONB", False),
        ("tool_calls", "JSONB", False),
        ("created_at", "TIMESTAMPTZ DEFAULT NOW()", False),
    ],
}


# ===========================================================================
# Index Definitions
# ===========================================================================

INDEX_DEFINITIONS = [
    # ── document_chunks ──
    "CREATE INDEX IF NOT EXISTS idx_dc_session ON document_chunks(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_dc_filepath ON document_chunks(filepath);",
    "CREATE INDEX IF NOT EXISTS idx_dc_file_type ON document_chunks(file_type);",
    "CREATE INDEX IF NOT EXISTS idx_dc_search_vector ON document_chunks USING GIN(search_vector);",

    # ── media_files ──
    "CREATE INDEX IF NOT EXISTS idx_mf_session ON media_files(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_mf_filepath ON media_files(filepath);",
    "CREATE INDEX IF NOT EXISTS idx_mf_search_vector ON media_files USING GIN(search_vector);",

    # ── structured_files ──
    "CREATE INDEX IF NOT EXISTS idx_sf_session ON structured_files(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_sf_filepath ON structured_files(filepath);",
    "CREATE INDEX IF NOT EXISTS idx_sf_search_vector ON structured_files USING GIN(search_vector);",

    # ── chat_messages ──
    "CREATE INDEX IF NOT EXISTS idx_cm_conversation ON chat_messages(conversation_id);",
    "CREATE INDEX IF NOT EXISTS idx_cm_session ON chat_messages(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_cm_created ON chat_messages(created_at);",

    # ── conversations ──
    "CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at DESC);",
]

VECTOR_INDEX_DEFINITIONS = [
    """CREATE INDEX IF NOT EXISTS idx_dc_embedding_hnsw
       ON document_chunks USING hnsw (embedding vector_cosine_ops)
       WITH (m = 16, ef_construction = 64);""",

    """CREATE INDEX IF NOT EXISTS idx_mf_embedding_hnsw
       ON media_files USING hnsw (transcript_embedding vector_cosine_ops)
       WITH (m = 16, ef_construction = 64);""",

    """CREATE INDEX IF NOT EXISTS idx_sf_embedding_hnsw
       ON structured_files USING hnsw (summary_embedding vector_cosine_ops)
       WITH (m = 16, ef_construction = 64);""",
]


# ===========================================================================
# Database Creation
# ===========================================================================

def _database_exists(admin_conn, db_name: str) -> bool:
    """Check whether a database exists on the PostgreSQL server."""
    cur = admin_conn.cursor()
    cur.execute(
        "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
        (db_name,),
    )
    exists = cur.fetchone() is not None
    cur.close()
    return exists


def _create_database(db_name: str):
    """
    Connect to the default 'postgres' database and CREATE DATABASE if needed.

    Uses ISOLATION_LEVEL_AUTOCOMMIT because CREATE DATABASE cannot run inside
    a transaction block.
    """
    logger.info("Checking if database '%s' exists...", db_name)

    admin_conn = None
    try:
        admin_conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USERNAME,
            password=PG_PASSWORD,
            dbname="postgres",          # connect to default admin database
            connect_timeout=10,
        )
        admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        if _database_exists(admin_conn, db_name):
            logger.info("Database '%s' already exists.", db_name)
        else:
            logger.info("Creating database '%s'...", db_name)
            cur = admin_conn.cursor()
            # Use psycopg2.sql to safely quote the database name
            cur.execute(
                psycopg2_sql.SQL("CREATE DATABASE {}").format(
                    psycopg2_sql.Identifier(db_name)
                )
            )
            cur.close()
            logger.info("Database '%s' created successfully.", db_name)

    except psycopg2.OperationalError as e:
        # If we can't even connect to 'postgres', the server is unreachable
        logger.error(
            "Cannot connect to PostgreSQL server at %s:%s to create database: %s",
            PG_HOST, PG_PORT, e,
        )
        raise RuntimeError(
            f"PostgreSQL server unreachable at {PG_HOST}:{PG_PORT}. "
            f"Ensure PostgreSQL is running and credentials are correct. "
            f"Original error: {e}"
        ) from e
    finally:
        if admin_conn:
            admin_conn.close()


# ===========================================================================
# Schema Migration — add missing columns to existing tables
# ===========================================================================

def _get_existing_columns(cur, table_name: str) -> set:
    """Return a set of column names that currently exist on a table."""
    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
          AND table_schema = 'public'
    """, (table_name,))
    return {row[0] for row in cur.fetchall()}


def _table_exists(cur, table_name: str) -> bool:
    """Check if a table exists in the public schema."""
    cur.execute("""
        SELECT 1 FROM information_schema.tables
        WHERE table_name = %s AND table_schema = 'public'
    """, (table_name,))
    return cur.fetchone() is not None


def _migrate_missing_columns(cur):
    """
    For each table that already exists, check for missing columns and
    add them via ALTER TABLE. This handles the case where tables were
    created under an older schema (v1) that lacked columns like
    session_id, search_vector, storage_path, etc.

    Generated columns (like search_vector tsvector) require special
    ALTER TABLE ... ADD COLUMN syntax.
    """
    for table_name in TABLE_CREATE_ORDER:
        if not _table_exists(cur, table_name):
            # Table doesn't exist yet — CREATE TABLE will handle it
            continue

        expected = EXPECTED_COLUMNS.get(table_name, [])
        if not expected:
            continue

        existing = _get_existing_columns(cur, table_name)
        missing = [(col, typedef, is_gen) for col, typedef, is_gen in expected
                    if col not in existing]

        if not missing:
            continue

        logger.info("  Migrating table '%s': adding %d missing column(s)...",
                     table_name, len(missing))

        for col_name, col_type, is_generated in missing:
            # Skip PRIMARY KEY columns — they can't be added after the fact
            if "PRIMARY KEY" in col_type.upper():
                logger.warning("    Skipping PK column '%s' on '%s' — cannot add PK via ALTER TABLE",
                               col_name, table_name)
                continue

            try:
                alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"
                cur.execute(alter_sql)
                logger.info("    Added column: %s.%s", table_name, col_name)
            except psycopg2.Error as col_err:
                # Column might already exist (race condition) or type conflict
                logger.warning("    Could not add column %s.%s: %s",
                               table_name, col_name, col_err)


# ===========================================================================
# Extension, Table, and Index Setup
# ===========================================================================

def _setup_schema(conn):
    """
    Create the pgvector extension, all tables, migrate missing columns,
    and create all indexes.

    Everything uses IF NOT EXISTS so it's fully idempotent.
    """
    conn.autocommit = True
    cur = conn.cursor()

    # 1. pgvector extension
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("  pgvector extension: ready")
    except psycopg2.Error as e:
        logger.error("Failed to create pgvector extension: %s", e)
        raise RuntimeError(
            "Could not create pgvector extension. Ensure the 'vector' extension "
            "is installed on your PostgreSQL server. On Ubuntu/Debian: "
            "apt install postgresql-16-pgvector (adjust version as needed). "
            f"Original error: {e}"
        ) from e

    # 2. Migrate missing columns on existing tables BEFORE creating tables
    #    (so that CREATE TABLE IF NOT EXISTS + ALTER TABLE cover all cases)
    try:
        _migrate_missing_columns(cur)
    except Exception as mig_err:
        logger.warning("  Column migration had issues: %s", mig_err)

    # 3. Tables (order matters for foreign keys)
    for table_name in TABLE_CREATE_ORDER:
        try:
            cur.execute(TABLE_DEFINITIONS[table_name])
            logger.info("  Table ready: %s", table_name)
        except psycopg2.Error as tbl_err:
            logger.warning("  Table creation note for '%s': %s", table_name, tbl_err)

    # 4. Run migration again after table creation to catch any tables
    #    that were just created but might still be missing columns
    #    (e.g., if the CREATE TABLE above was a no-op for an existing table)
    try:
        _migrate_missing_columns(cur)
    except Exception as mig_err:
        logger.warning("  Post-creation migration note: %s", mig_err)

    # 5. B-tree and GIN indexes
    for idx_sql in INDEX_DEFINITIONS:
        try:
            cur.execute(idx_sql)
        except psycopg2.Error as idx_err:
            logger.warning("  Index note: %s", idx_err)

    # 6. HNSW vector indexes
    for vidx_sql in VECTOR_INDEX_DEFINITIONS:
        try:
            cur.execute(vidx_sql)
        except psycopg2.Error as vidx_err:
            logger.warning("  Vector index note: %s", vidx_err)

    logger.info("  All indexes: ready")
    cur.close()


# ===========================================================================
# Public API
# ===========================================================================

def ensure_db_ready():
    """
    Ensure the database, extension, tables, and indexes all exist.

    Thread-safe. Only runs the full setup once per process. Subsequent calls
    are no-ops and return immediately.

    Call this before any database operation, or simply use get_connection()
    which calls it automatically.
    """
    global _initialized

    if _initialized:
        return

    with _init_lock:
        # Double-check after acquiring lock (another thread may have finished)
        if _initialized:
            return

        logger.info("Initializing database infrastructure...")

        # Step 1: Create the database if it doesn't exist
        _create_database(PG_DATABASE)

        # Step 2: Connect to the target database and set up schema
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USERNAME,
            password=PG_PASSWORD,
            dbname=PG_DATABASE,
            connect_timeout=10,
        )
        try:
            _setup_schema(conn)
        finally:
            conn.close()

        _initialized = True
        logger.info("Database infrastructure ready.")


def get_connection(**kwargs):
    """
    Return a psycopg2 connection to the configured database.

    Automatically calls ensure_db_ready() on the first invocation so the
    caller never has to worry about database/table existence.

    Any extra keyword arguments are forwarded to psycopg2.connect()
    (e.g. cursor_factory, connect_timeout).
    """
    ensure_db_ready()

    defaults = {
        "host": PG_HOST,
        "port": PG_PORT,
        "user": PG_USERNAME,
        "password": PG_PASSWORD,
        "dbname": PG_DATABASE,
        "connect_timeout": 10,
    }
    defaults.update(kwargs)
    return psycopg2.connect(**defaults)


def get_connection_from_state(state: dict):
    """
    Return a connection using credentials from the pipeline state dict.

    This is used by pipeline nodes which may receive PG credentials via
    the LangGraph state rather than from config directly. Still calls
    ensure_db_ready() first to guarantee infrastructure exists.
    """
    ensure_db_ready()

    return psycopg2.connect(
        host=state.get("pg_host", PG_HOST),
        port=state.get("pg_port", PG_PORT),
        user=state.get("pg_username", PG_USERNAME),
        password=state.get("pg_password", PG_PASSWORD),
        dbname=state.get("pg_database", PG_DATABASE),
        connect_timeout=10,
    )


def reset_init_flag():
    """
    Reset the initialization flag. Only useful for testing or if you need
    to force re-initialization after changing config at runtime.
    """
    global _initialized
    with _init_lock:
        _initialized = False
