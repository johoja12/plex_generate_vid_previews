from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel
from enum import Enum

class PreviewStatus(str, Enum):
    MISSING = "missing"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SLOW_FAILED = "slow_failed"
    IGNORED = "ignored"
    MEDIA_MISSING = "media_missing"  # Media file doesn't exist (broken symlink, deleted file)

class MediaType(str, Enum):
    MOVIE = "movie"
    EPISODE = "episode"

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str

class AppSettings(SQLModel, table=True):
    id: int = Field(default=1, primary_key=True)
    plex_url: Optional[str] = None
    plex_token: Optional[str] = None
    plex_server_name: Optional[str] = None
    plex_client_identifier: Optional[str] = None # Plex server identifier
    admin_password_hash: Optional[str] = None

    # We can store other config overrides here too if we want
    gpu_threads: int = 1
    cpu_threads: int = 1
    scheduler_loop_interval: int = 5 # Default to 5 seconds
    sync_interval: int = 21600  # Sync interval in seconds (default: 6 hours)
    last_sync_time: Optional[datetime] = None
    last_sync_summary: Optional[str] = None  # JSON summary of last sync (items added/updated/deleted)
    queue_paused: bool = False  # Persistent paused/resumed state for queue processing

    # Mount protection settings
    mount_check_enabled: bool = True  # Enable mount validation before sync/processing
    mount_check_paths: Optional[str] = None  # Comma-separated paths to check (e.g., "/mnt/media/.mounted,/data/.marker")
    mount_failure_threshold: float = 50.0  # If more than this % of items become missing, pause and alert (default: 50%)

    # Multi-user priority settings
    enable_multi_user_priority: bool = False  # Enable priority detection for all users on the server
    priority_history_limit: int = 50  # Number of recently watched items to check per user (default: 50)

    # Data rate limiting
    data_limit_gb_per_hour: float = Field(default=0.0) # 0.0 means no limit
    total_bytes_processed_hour: int = Field(default=0)
    hour_start_time: datetime = Field(default_factory=datetime.utcnow)
    rate_limit_exempt_paths: Optional[str] = None  # Comma-separated paths exempt from rate limiting (e.g., "/mnt/remote/nzbdav")

    # Sync settings
    use_database_sync: bool = True  # Use direct database queries for sync instead of Plex API (faster)

    updated_at: datetime = Field(default_factory=datetime.utcnow)

class MediaItem(SQLModel, table=True):
    id: int = Field(primary_key=True, description="Plex Rating Key")
    title: str
    media_type: MediaType
    library_name: str
    original_available_at: Optional[datetime] = None # For sorting by release date
    added_at: datetime = Field(default_factory=datetime.utcnow) # When added to Plex

    # Processing State
    status: PreviewStatus = Field(default=PreviewStatus.MISSING)
    progress: int = Field(default=0)
    current_processing_bundle_hash: Optional[str] = None  # To track which part is currently processing
    queue_order: int = Field(default=0) # For manual ordering
    is_priority: bool = Field(default=False) # Priority items (on deck, recently played, adjacent episodes)
    avg_speed: Optional[str] = None # Average processing speed (e.g., "1.23x")

    # Metadata
    file_path: Optional[str] = None  # Primary file path (for backwards compatibility)
    duration: Optional[int] = None # In milliseconds
    bundle_hash: Optional[str] = None  # Primary bundle hash (for backwards compatibility)
    bif_path: Optional[str] = None # Path to the generated BIF file (primary)
    media_parts_info: Optional[str] = None  # JSON array of all media parts/versions

    # Error tracking
    error_message: Optional[str] = None # Failure reason for failed items

    updated_at: datetime = Field(default_factory=datetime.utcnow)