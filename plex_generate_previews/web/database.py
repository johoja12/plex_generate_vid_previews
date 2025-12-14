from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy import event, text
import os

# Define database URL
# Check for DB_PATH environment variable, default to local file
sqlite_file_name = os.environ.get("DB_PATH", "plex_previews.db")
DATABASE_URL = f"sqlite:///{sqlite_file_name}"

# Increase timeout to reduce locking errors (30 seconds for high concurrency)
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False, "timeout": 30})

def setup_sqlite(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;") # Optional but good for performance
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

def get_session():
    with Session(engine) as session:
        yield session