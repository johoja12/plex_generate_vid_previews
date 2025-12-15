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
    queue_order: int = Field(default=0) # For manual ordering
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