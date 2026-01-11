from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy import event, text
import os

# Define database URL
# Check for DB_PATH environment variable, default to local file
sqlite_file_name = os.environ.get("DB_PATH", "plex_previews.db")
DATABASE_URL = f"sqlite:///{sqlite_file_name}"

# Increase timeout to reduce locking errors (60 seconds for high concurrency with batching)
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={
        "check_same_thread": False,
        "timeout": 60.0,  # Increased timeout
        "isolation_level": None  # Autocommit mode for better concurrency
    },
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600  # Recycle connections every hour
)

def setup_sqlite(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    # WAL mode allows concurrent reads and writes
    cursor.execute("PRAGMA journal_mode=WAL;")
    # NORMAL is faster than FULL, still safe with WAL
    cursor.execute("PRAGMA synchronous=NORMAL;")
    # Increase cache size for better performance (in KB, negative = KB)
    cursor.execute("PRAGMA cache_size=-64000;")  # 64MB cache
    # Optimize for concurrent access
    cursor.execute("PRAGMA busy_timeout=60000;")  # 60 second busy timeout
    # Use memory for temp storage (faster)
    cursor.execute("PRAGMA temp_store=MEMORY;")
    cursor.close()

event.listen(engine, "connect", setup_sqlite)

def create_db_and_tables():
    # Ensure directory exists if it's a path
    db_dir = os.path.dirname(sqlite_file_name)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    SQLModel.metadata.create_all(engine)
    
    # Migration: Add admin_password_hash if missing
    # This is a simple migration for SQLite since SQLModel doesn't auto-migrate columns
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE appsettings ADD COLUMN admin_password_hash VARCHAR"))
    except Exception:
        # Column likely exists
        pass

    # Migration: Add queue_order if missing
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE mediaitem ADD COLUMN queue_order INTEGER DEFAULT 0"))
    except Exception:
        pass

    # Migration: Add bundle_hash if missing
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE mediaitem ADD COLUMN bundle_hash VARCHAR"))
    except Exception:
        pass

    # Migration: Add last_sync_time to AppSettings
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE appsettings ADD COLUMN last_sync_time DATETIME"))
    except Exception:
        pass

    # Migration: Add plex_client_identifier to AppSettings
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE appsettings ADD COLUMN plex_client_identifier VARCHAR"))
    except Exception:
        pass

    # Migration: Add scheduler_loop_interval to AppSettings
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE appsettings ADD COLUMN scheduler_loop_interval INTEGER DEFAULT 5"))
    except Exception:
        pass

    # Migration: Add sync_interval to AppSettings (default: 6 hours = 21600 seconds)
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE appsettings ADD COLUMN sync_interval INTEGER DEFAULT 21600"))
    except Exception:
        pass

    # Migration: Add error_message to MediaItem
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE mediaitem ADD COLUMN error_message VARCHAR"))
    except Exception:
        pass

    # Migration: Add bif_path to MediaItem
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE mediaitem ADD COLUMN bif_path VARCHAR"))
    except Exception:
        pass

    # Migration: Add avg_speed to MediaItem
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE mediaitem ADD COLUMN avg_speed VARCHAR"))
    except Exception:
        pass

    # Migration: Add media_parts_info to MediaItem
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE mediaitem ADD COLUMN media_parts_info VARCHAR"))
    except Exception:
        pass

    # Migration: Add is_priority to MediaItem
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE mediaitem ADD COLUMN is_priority INTEGER DEFAULT 0"))
    except Exception:
        pass

    # Migration: Add current_processing_bundle_hash to MediaItem
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE mediaitem ADD COLUMN current_processing_bundle_hash VARCHAR"))
    except Exception:
        pass

    # Migration: Add queue_paused to AppSettings
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE appsettings ADD COLUMN queue_paused INTEGER DEFAULT 0"))
    except Exception:
        pass

    # Migration: Add mount protection settings to AppSettings
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE appsettings ADD COLUMN mount_check_enabled INTEGER DEFAULT 1"))
    except Exception:
        pass

    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE appsettings ADD COLUMN mount_check_paths VARCHAR"))
    except Exception:
        pass

    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE appsettings ADD COLUMN mount_failure_threshold REAL DEFAULT 50.0"))
    except Exception:
        pass

    # Migration: Add last_sync_summary to AppSettings
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE appsettings ADD COLUMN last_sync_summary VARCHAR"))
    except Exception:
        pass

    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE appsettings ADD COLUMN enable_multi_user_priority INTEGER DEFAULT 0"))
    except Exception:
        pass

    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE appsettings ADD COLUMN priority_history_limit INTEGER DEFAULT 50"))
    except Exception:
        pass

    # Migration: Add data rate limiting settings to AppSettings
    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE appsettings ADD COLUMN data_limit_gb_per_hour REAL DEFAULT 0.0"))
    except Exception:
        pass

    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE appsettings ADD COLUMN total_bytes_processed_hour BIGINT DEFAULT 0"))
    except Exception:
        pass

    try:
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE appsettings ADD COLUMN hour_start_time DATETIME"))
    except Exception:
        pass

def get_session():
    with Session(engine) as session:
        yield session