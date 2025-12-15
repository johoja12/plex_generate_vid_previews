from typing import Protocol, Any, List, Optional
from dataclasses import dataclass

@dataclass
class ProgressData:
    progress_percent: float
    speed: str
    task_title: str
    is_busy: bool
    current_duration: float
    total_duration: float
    remaining_time: float
    frame: int = 0
    fps: float = 0
    q: float = 0
    size: int = 0
    time_str: str = ""
    bitrate: float = 0

class ProgressManager(Protocol):
    def init_workers(self, workers: List[Any]) -> None:
        """Initialize progress tracking for a list of workers."""
        ...
        
    def update_worker(self, worker_id: int, data: ProgressData) -> None:
        """Update the progress display/state for a specific worker."""
        ...
        
    def update_main_progress(self, completed: int, total: int) -> None:
        """Update the overall progress."""
        ...
        
    def cleanup_workers(self) -> None:
        """Cleanup worker progress displays."""
        ...

class NullProgressManager:
    """A no-op progress manager."""
    def init_workers(self, workers: List[Any]) -> None: pass
    def update_worker(self, worker_id: int, data: ProgressData) -> None: pass
    def update_main_progress(self, completed: int, total: int) -> None: pass
    def cleanup_workers(self) -> None: pass
