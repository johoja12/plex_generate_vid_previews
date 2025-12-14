"""
Worker classes for processing media items using threading.

Provides Worker and WorkerPool classes that use threading instead of
multiprocessing for better simplicity and performance with FFmpeg tasks.
"""

import re
import threading
import time
import queue
from functools import partial
from typing import List, Optional, Any, Tuple
from loguru import logger

from .config import Config
from .media_processing import process_item, CodecNotSupportedError, ItemNotFoundError, SlowProcessingError
from .utils import format_display_title
from .progress_reporter import ProgressManager


class Worker:
    """Represents a worker thread for processing media items."""
    
    def __init__(self, worker_id: int, worker_type: str, gpu: Optional[str] = None, 
                 gpu_device: Optional[str] = None, gpu_index: Optional[int] = None, 
                 gpu_name: Optional[str] = None):
        """
        Initialize a worker.
        
        Args:
            worker_id: Unique identifier for this worker
            worker_type: 'GPU' or 'CPU'
            gpu: GPU type for acceleration
            gpu_device: GPU device path
            gpu_index: Index of the assigned GPU hardware
            gpu_name: Human-readable GPU name for display
        """
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.gpu = gpu
        self.gpu_device = gpu_device
        self.gpu_index = gpu_index
        self.gpu_name = gpu_name
        
        # Task state
        self.is_busy = False
        self.current_thread = None
        self.current_task = None
        self.failed_state = False
        self.error_message = None  # Error message for failed tasks
        
        # Progress tracking
        self.progress_percent = 0
        self.speed = "0.0x"
        self.avg_speed = "0.0x"  # 5-minute average speed
        self.current_duration = 0.0
        self.total_duration = 0.0
        self.remaining_time = 0.0  # Remaining time calculated from FFmpeg data
        self.task_title = ""
        self.display_title = ""
        self.media_title = ""
        self.media_type = ""
        self.media_file = ""  # Actual file path being processed
        self.title_max_width = 20
        self.progress_task_id = None
        self.ffmpeg_started = False  # Track if FFmpeg has started outputting progress
        self.task_start_time = 0  # Track when task started
        
        # FFmpeg data fields for display
        self.frame = 0
        self.fps = 0
        self.q = 0
        self.size = 0
        self.time_str = "00:00:00.00"
        self.bitrate = 0
        
        # Track last update to avoid unnecessary updates
        self.last_progress_percent = -1
        self.last_speed = ""
        self.last_update_time = 0
        
        # Track verbose logging
        self.last_verbose_log_time = 0

        # Statistics
        self.completed = 0
        self.failed = 0
        self.skipped = 0  # Items not found in Plex (404 errors)
    
    def is_available(self) -> bool:
        """Check if this worker is available for a new task."""
        return not self.is_busy
    
    def _format_gpu_name_for_display(self) -> str:
        """Format GPU name for consistent display width."""
        if not self.gpu_name:
            return f"GPU {self.gpu_index}"
        
        # If already short enough, pad to 10 characters
        if len(self.gpu_name) <= 10:
            return self.gpu_name.ljust(10)[:10]
        
        # Dictionary of GPU name patterns and their shortened forms
        # Pattern matching rules: (pattern, replacement or extraction function)
        patterns = [
            (r'.*TITAN.*RTX.*', lambda m: "TITAN RTX"),  # TITAN RTX -> "TITAN RTX"
            (r'.*RTX\s*(\d+).*', lambda m: f"RTX{m.group(1)}"[:8]),  # Extract RTX number
            (r'.*GTX\s*(\d+).*', lambda m: f"GTX{m.group(1)}"[:8]),  # Extract GTX number
            (r'.*GeForce\s+([A-Z0-9\s]+).*', lambda m: m.group(1).strip()[:8]),  # Extract GeForce model
            (r'.*TITAN.*', lambda m: "TITAN"),  # TITAN (without RTX)
            (r'.*Intel.*', lambda m: "Intel"),  # Intel GPUs
            (r'.*AMD.*', lambda m: "AMD"),  # AMD GPUs
        ]
        
        # Try each pattern in order
        for pattern, replacement in patterns:
            match = re.search(pattern, self.gpu_name)
            if match:
                result = replacement(match) if callable(replacement) else replacement
                return result.ljust(10)[:10]
        
        # Fallback: truncate to 8 characters
        return self.gpu_name[:8].ljust(10)[:10]
    
    def _format_idle_description(self) -> str:
        """Format idle description for display."""
        if self.worker_type == 'GPU':
            gpu_display = self._format_gpu_name_for_display()
            return f"[{gpu_display}]: Idle - Waiting for task..."
        return f"[CPU      ]: Idle - Waiting for task..."
    
    def assign_task(self, item_key: str, config: Config, plex, progress_callback=None, 
                   media_title: str = "", media_type: str = "", title_max_width: int = 20, 
                   cpu_fallback_queue=None) -> None:
        """
        Assign a new task to this worker.
        
        Args:
            item_key: Plex media item key to process
            config: Configuration object
            plex: Plex server instance
            progress_callback: Callback function for progress updates
            media_title: Media title for display
            media_type: Media type ('episode' or 'movie')
            title_max_width: Maximum width for title display
            cpu_fallback_queue: Optional queue to add task to if codec error occurs (GPU workers only)
        """
        if self.is_busy:
            raise RuntimeError(f"Worker {self.worker_id} is already busy")
        
        # Reset all progress tracking to ensure clean state
        self.is_busy = True
        self.current_task = item_key
        self.failed_state = False
        self.error_message = None
        self.media_title = media_title
        self.media_type = media_type
        self.media_file = ""  # Will be populated by progress callback
        self.title_max_width = title_max_width
        self.display_title = format_display_title(media_title, media_type, title_max_width)
        # Show GPU name in display for GPU workers, show CPU identifier for CPU workers
        if self.worker_type == 'GPU':
            gpu_display = self._format_gpu_name_for_display()
            self.task_title = f"[{gpu_display}]: {self.display_title}"
        else:
            self.task_title = f"[CPU      ]: {self.display_title}"
        self.progress_percent = 0
        self.speed = "0.0x"
        self.current_duration = 0.0
        self.total_duration = 0.0
        self.remaining_time = 0.0
        self.ffmpeg_started = False
        self.task_start_time = time.time()
        
        # Reset FFmpeg data fields
        self.frame = 0
        self.fps = 0
        self.q = 0
        self.size = 0
        self.time_str = "00:00:00.00"
        self.bitrate = 0
        
        # Reset tracking variables for clean state
        self.last_progress_percent = -1
        self.last_speed = ""
        self.last_update_time = 0
        self.last_verbose_log_time = 0
        
        # Start processing in background thread
        self.current_thread = threading.Thread(
            target=self._process_item, 
            args=(item_key, config, plex, progress_callback, cpu_fallback_queue),
            daemon=True
        )
        self.current_thread.start()
    
    def _process_item(self, item_key: str, config: Config, plex, progress_callback=None, cpu_fallback_queue=None) -> None:
        """
        Process a media item in the background thread.
        
        Args:
            item_key: Plex media item key
            config: Configuration object
            plex: Plex server instance
            progress_callback: Callback function for progress updates
            cpu_fallback_queue: Optional queue to add task to if codec error occurs (GPU workers only)
        """
        # Use file path if available, otherwise fall back to title or item_key
        display_name = self.media_file if self.media_file else (self.media_title if self.media_title else item_key)
        
        try:
            process_item(item_key, self.gpu, self.gpu_device, config, plex, progress_callback)

            # Ensure 100% progress is reported upon success
            if progress_callback:
                progress_callback(100, self.total_duration, self.total_duration, speed="Finished")

            # Mark as completed immediately (thread will finish after this)
            self.completed += 1
        except ItemNotFoundError as e:
            # Item not found in Plex (404) - mark as failed in DB to prevent re-queuing
            error_msg = f"Not found in Plex: {str(e)}"
            logger.info(f"Worker {self.worker_id} skipping {display_name}: item not found in Plex")
            self.error_message = error_msg
            if progress_callback:
                # Mark as failed so scheduler won't re-queue this item
                progress_callback(0, 0, 0, speed="Skipped", failed=True, error_message=error_msg)
            self.skipped += 1
        except CodecNotSupportedError as e:
            # Codec not supported by GPU - re-queue for CPU worker
            if self.worker_type == 'GPU':
                logger.warning(f"GPU Worker {self.worker_id} detected unsupported codec for {display_name}; handing off to CPU worker")
                # Add to fallback queue for CPU worker processing (multiple CPU workers can compete for items)
                if cpu_fallback_queue is not None and config.cpu_threads > 0:
                    # Preserve media info (set during assign_task)
                    try:
                        cpu_fallback_queue.put((item_key, self.media_title, self.media_type))
                        logger.debug(f"Added {display_name} to CPU fallback queue")
                    except Exception as queue_error:
                        logger.error(f"Failed to add {item_key} to fallback queue: {queue_error}")
                        error_msg = f"Failed to add to fallback queue: {str(queue_error)}"
                        self.error_message = error_msg
                        if progress_callback:
                            progress_callback(0, 0, 0, speed="Failed", failed=True, error_message=error_msg)
                        self.failed += 1
                else:
                    if config.cpu_threads == 0:
                        error_msg = f"Codec not supported by GPU, CPU threads disabled: {str(e)}"
                        logger.warning(f"Codec not supported by GPU, but CPU threads are disabled (CPU_THREADS=0); skipping {display_name}")
                        self.error_message = error_msg
                        if progress_callback:
                            progress_callback(0, 0, 0, speed="Failed", failed=True, error_message=error_msg)
                    self.failed += 1
                # Mark as completed from GPU worker perspective (task will be handled by CPU)
                self.completed += 1
            else:
                # CPU worker received codec error - this is unexpected, treat as failure
                error_msg = f"CPU codec error (file may be corrupted): {str(e)}"
                logger.error(f"CPU Worker {self.worker_id} encountered codec error for {display_name}: {e}")
                logger.error("Codec errors should not occur on CPU workers - file may be corrupted")
                self.error_message = error_msg
                if progress_callback:
                    progress_callback(0, 0, 0, speed="Failed", failed=True, error_message=error_msg)
                self.failed += 1
        except SlowProcessingError as e:
            # Processing too slow (< 1x for > 5 minutes) - mark as failed
            error_msg = "Processing too slow (< 1x for > 5 minutes)"
            logger.warning(f"Worker {self.worker_id} marking {display_name} as failed: {error_msg}")
            self.error_message = error_msg
            if progress_callback:
                progress_callback(0, 0, 0, speed="Too Slow", failed=True, error_message=error_msg)
            self.failed += 1
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Worker {self.worker_id} failed to process {display_name}: {e}")
            logger.debug(f"Traceback for {display_name}:", exc_info=True)
            self.error_message = error_msg
            if progress_callback:
                progress_callback(0, 0, 0, speed="Failed", failed=True, error_message=error_msg)
            self.failed += 1
    
    def check_completion(self) -> bool:
        """
        Check if this worker has completed its current task.

        Returns:
            bool: True if task completed, False if still running
        """
        if not self.is_busy:
            return False  # Worker is available, not completing

        if self.current_thread and not self.current_thread.is_alive():
            # Thread finished - mark as completed but keep task info for final update
            # Don't clear current_task yet - it's needed for the final progress update
            self.is_busy = False
            # Note: current_task will be cleared after final progress update in process_items loop
            return True

        return False
    
    def get_progress_data(self) -> dict:
        """Get current progress data for main thread."""
        return {
            'progress_percent': self.progress_percent,
            'speed': self.speed,
            'task_title': self.task_title,
            'is_busy': self.is_busy,
            'current_duration': self.current_duration,
            'total_duration': self.total_duration,
            'remaining_time': self.remaining_time,
            'worker_id': self.worker_id,  # Add worker ID for debugging
            'worker_type': self.worker_type,  # Add worker type for debugging
            'item_key': self.current_task,
            'failed': self.failed_state,
            'error_message': self.error_message,  # Error message for failed tasks
            'media_file': self.media_file,  # Media file path being processed
            # FFmpeg data for display
            'frame': self.frame,
            'fps': self.fps,
            'q': self.q,
            'size': self.size,
            'time_str': self.time_str,
            'bitrate': self.bitrate
        }
    
    def shutdown(self) -> None:
        """Shutdown the worker gracefully."""
        if self.current_thread and self.current_thread.is_alive():
            # Wait for current task to complete (with timeout)
            self.current_thread.join(timeout=5)
    
    @staticmethod
    def find_available(workers: List['Worker']) -> Optional['Worker']:
        """
        Find the first available worker.
        
        GPU workers are prioritized (they come first in the array).
        
        Args:
            workers: List of Worker instances
            
        Returns:
            Worker: First available worker, or None if all are busy
        """
        for worker in workers:
            if worker.is_available():
                return worker
        return None


class WorkerPool:
    """Manages a pool of workers for processing media items."""
    
    def __init__(self, gpu_workers: int, cpu_workers: int, selected_gpus: List[Tuple[str, str, dict]]):
        """
        Initialize worker pool.
        
        Args:
            gpu_workers: Number of GPU workers to create
            cpu_workers: Number of CPU workers to create
            selected_gpus: List of (gpu_type, gpu_device, gpu_info) tuples for GPU workers
        """
        self.workers = []
        self._progress_lock = threading.Lock()  # Thread-safe progress updates
        self.cpu_fallback_queue = queue.Queue()  # Thread-safe queue for CPU-only tasks (codec fallback)
        
        # Add GPU workers first (prioritized) with round-robin GPU assignment
        for i in range(gpu_workers):
            # selected_gpus is guaranteed to be non-empty if gpu_workers > 0
            # because detect_and_select_gpus() exits with error if no GPUs detected
            gpu_index = i % len(selected_gpus)
            gpu_type, gpu_device, gpu_info = selected_gpus[gpu_index]
            gpu_name = gpu_info.get('name', f'{gpu_type} GPU')
            
            worker = Worker(i, 'GPU', gpu_type, gpu_device, gpu_index, gpu_name)
            self.workers.append(worker)
            
            logger.info(f'GPU Worker {i} assigned to GPU {gpu_index} ({gpu_name})')
        
        # Add CPU workers
        for i in range(cpu_workers):
            self.workers.append(Worker(i + gpu_workers, 'CPU'))
        
        logger.info(f'Initialized {len(self.workers)} workers: {gpu_workers} GPU + {cpu_workers} CPU')
    
    def has_busy_workers(self) -> bool:
        """Check if any workers are currently busy."""
        return any(worker.is_busy for worker in self.workers)
    
    def has_available_workers(self) -> bool:
        """Check if any workers are available for new tasks."""
        return any(worker.is_available() for worker in self.workers)
    
    def _find_available_worker(self, cpu_only: bool = False) -> Optional['Worker']:
        """
        Find an available worker.
        
        Args:
            cpu_only: If True, only look for CPU workers
            
        Returns:
            First available worker matching criteria, or None
        """
        if cpu_only:
            for worker in self.workers:
                if worker.worker_type == 'CPU' and worker.is_available():
                    return worker
            return None
        return Worker.find_available(self.workers)
    
    def _get_plex_media_info(self, plex, item_key: str) -> Tuple[str, str]:
        """
        Re-query Plex for media information if not available.
        
        Returns:
            Tuple of (media_title, media_type)
        """
        try:
            from .plex_client import retry_plex_call
            data = retry_plex_call(plex.query, item_key)
            if data is not None:
                video_element = data.find('Video') or data.find('Directory')
                if video_element is not None:
                    return (video_element.get('title', 'Unknown (fallback)'), 
                           video_element.tag.lower())
        except Exception as e:
            logger.debug(f"Could not re-query Plex for {item_key}: {e}")
        return ('Unknown (fallback)', 'unknown')
    
    def _assign_fallback_task(self, worker: 'Worker', config: Config, plex,
                              title_max_width: int) -> bool:
        """
        Assign a task from fallback queue to a CPU worker.

        Returns:
            True if task was assigned, False if queue was empty
        """
        try:
            fallback_item = self.cpu_fallback_queue.get_nowait()
            item_key, media_title, media_type = fallback_item

            # Check if any other worker is already processing this item
            for other_worker in self.workers:
                if other_worker != worker and other_worker.current_task == item_key:
                    logger.warning(f"Skipping duplicate fallback assignment: {media_title} (item {item_key}) already being processed by worker {other_worker.worker_id}")
                    # Put the item back in the queue since we didn't assign it
                    self.cpu_fallback_queue.put(fallback_item)
                    return False

            # Re-query Plex for media info if not available
            if media_title is None or media_type is None:
                media_title, media_type = self._get_plex_media_info(plex, item_key)

            progress_callback = partial(self._update_worker_progress, worker)
            worker.assign_task(
                item_key, config, plex,
                progress_callback=progress_callback,
                media_title=media_title,
                media_type=media_type,
                title_max_width=title_max_width,
                cpu_fallback_queue=None
            )
            return True
        except queue.Empty:
            return False
    
    def _assign_main_queue_task(self, worker: 'Worker', media_queue: List[tuple],
                                config: Config, plex, title_max_width: int) -> tuple:
        """
        Assign a task from main queue to a worker.

        Returns:
            Tuple of (assigned: bool, queue_empty: bool)
            - assigned: True if task was successfully assigned
            - queue_empty: True if queue is empty (should break assignment loop)
        """
        if not media_queue:
            return (False, True)  # Not assigned, queue empty

        item_key, media_title, media_type = media_queue.pop(0)

        # Check if any other worker is already processing this item
        for other_worker in self.workers:
            if other_worker != worker and other_worker.current_task == item_key:
                logger.warning(f"Skipping duplicate assignment: {media_title} (item {item_key}) already being processed by worker {other_worker.worker_id}")
                # Put item back at end of queue to retry later
                media_queue.append((item_key, media_title, media_type))
                return (False, False)  # Not assigned, but queue not empty - try next worker

        progress_callback = partial(self._update_worker_progress, worker)
        cpu_fallback_queue = self.cpu_fallback_queue if worker.worker_type == 'GPU' else None

        worker.assign_task(
            item_key, config, plex,
            progress_callback=progress_callback,
            media_title=media_title,
            media_type=media_type,
            title_max_width=title_max_width,
            cpu_fallback_queue=cpu_fallback_queue
        )
        return (True, False)  # Assigned successfully
    
    def _check_fallback_queue_empty(self) -> bool:
        """
        Check if fallback queue is empty without consuming items.
        
        Returns:
            True if queue is empty, False if it has items
        """
        try:
            test_item = self.cpu_fallback_queue.get_nowait()
            self.cpu_fallback_queue.put(test_item)
            return False
        except queue.Empty:
            return True
    
    def process_items(self, media_items: List[tuple], config: Config, plex, progress_manager: ProgressManager, title_max_width: int = 20, library_name: str = "", stop_condition=None, fetch_more_items=None) -> None:
        """
        Process all media items using available workers.

        Uses dynamic task assignment - workers pull tasks as they become available.

        Args:
            media_items: List of tuples (key, title, media_type) to process
            config: Configuration object
            plex: Plex server instance
            progress_manager: ProgressManager instance for handling progress updates
            title_max_width: Maximum width for title display
            library_name: Name of the library section being processed
            stop_condition: Optional callable returning True if processing should stop
            fetch_more_items: Optional callable that returns List[tuple] of more items to process
        """
        media_queue = list(media_items)  # Copy the list
        completed_tasks = 0
        total_items = len(media_items)
        last_overall_progress_log = time.time()
        
        # Use provided title width for display formatting
        library_prefix = f"[{library_name}] " if library_name else ""
        
        logger.info(f'Processing {total_items} items with {len(self.workers)} workers')
        
        # Initialize worker progress tracking
        progress_manager.init_workers(self.workers)
        
        # Process all items
        # Continue while we have items in main queue, fallback queue, or busy workers
        # Exit conditions: main queue empty, all items processed, no busy workers, fallback queue empty
        while True:
            # Check for completed tasks and update progress
            for worker in self.workers:
                if worker.check_completion():
                    completed_tasks += 1
                    # Update main progress
                    progress_manager.update_main_progress(completed_tasks, total_items)

                    # Send final progress update with 100% if not already sent
                    # Worker is no longer busy but still has task info for final update
                    with self._progress_lock:
                        if worker.current_task:
                            # Send final update only if task didn't fail
                            if worker.failed_state:
                                # Task failed - ensure failed state is communicated
                                final_progress_data = worker.get_progress_data()
                                final_progress_data['failed'] = True
                                final_progress_data['is_busy'] = False
                                progress_manager.update_worker(worker.worker_id, final_progress_data)
                                logger.debug(f"Sent final failed update for worker {worker.worker_id}, item {worker.current_task}")
                            elif worker.progress_percent >= 99:
                                # Task succeeded - send 100% completion
                                final_progress_data = worker.get_progress_data()
                                final_progress_data['progress_percent'] = 100
                                final_progress_data['is_busy'] = False  # Important: worker is done but has data
                                progress_manager.update_worker(worker.worker_id, final_progress_data)
                                logger.debug(f"Sent final 100% progress update for worker {worker.worker_id}, item {worker.current_task}")
                        # Now clear the task after final update
                        worker.current_task = None

                # Update worker progress display with thread-safe access
                current_time = time.time()

                # Use thread-safe access to worker progress data
                with self._progress_lock:
                    progress_data = worker.get_progress_data()
                    is_busy = worker.is_busy
                    ffmpeg_started = worker.ffmpeg_started

                if is_busy:
                    # Update busy worker only if progress or speed changed and enough time has passed
                    should_update = (
                        (progress_data['progress_percent'] != worker.last_progress_percent or 
                         progress_data['speed'] != worker.last_speed or
                         not ffmpeg_started) and
                        (current_time - worker.last_update_time > 0.05)  # Throttle to 20fps for stability
                    )
                    
                    if should_update:
                        progress_manager.update_worker(worker.worker_id, progress_data)
                        worker.last_progress_percent = progress_data['progress_percent']
                        worker.last_speed = progress_data['speed']
                        worker.last_update_time = current_time
                else:
                    # Update idle worker only if it was previously busy
                    if worker.last_progress_percent != -1:
                        # Create generic idle data
                        idle_data = {
                            'progress_percent': 0,
                            'speed': "0.0x",
                            'task_title': worker._format_idle_description(),
                            'is_busy': False,
                            'current_duration': 0,
                            'total_duration': 0,
                            'remaining_time': 0,
                            'frame': 0, 'fps': 0, 'q': 0, 'size': 0, 
                            'time_str': "", 'bitrate': 0
                        }
                        progress_manager.update_worker(worker.worker_id, idle_data)
                        worker.last_progress_percent = -1
                        worker.last_speed = ""
            
            # Log overall progress every 5 seconds
            current_time = time.time()
            if current_time - last_overall_progress_log >= 5.0:
                progress_percent = int((completed_tasks / total_items) * 100) if total_items > 0 else 0
                logger.info(f"Processing progress {library_prefix}{completed_tasks}/{total_items} ({progress_percent}%) completed")

                # Log which workers are busy and what they're processing (for debugging stuck workers)
                busy_workers = [w for w in self.workers if w.is_busy]
                if busy_workers:
                    logger.debug(f"Busy workers: {len(busy_workers)}/{len(self.workers)}")
                    for worker in busy_workers:
                        with self._progress_lock:
                            avg_speed_display = f", avg: {worker.avg_speed}" if worker.avg_speed and worker.avg_speed != "0.0x" else ""
                            task_info = f"{worker.worker_id}: {worker.task_title} ({worker.progress_percent}%{avg_speed_display})"
                            if worker.current_thread and worker.current_thread.is_alive():
                                logger.debug(f"  {task_info} [thread alive]")
                            else:
                                logger.warning(f"  {task_info} [thread DEAD but still marked busy]")

                last_overall_progress_log = current_time
            
            # Assign new tasks to available workers
            # Prioritize fallback queue for CPU workers, then assign from main queue
            # Only assign if not stopped
            if not (stop_condition and stop_condition()):
                attempted_workers = 0
                max_attempts = len(self.workers) + 1  # Try all workers plus one for fallback queue

                while attempted_workers < max_attempts:
                    # If main queue is empty, only look for CPU workers (to process fallback queue)
                    cpu_only = not media_queue
                    available_worker = self._find_available_worker(cpu_only=cpu_only)
                    if not available_worker:
                        break

                    # For CPU workers, try fallback queue first (codec error fallback)
                    if available_worker.worker_type == 'CPU':
                        if self._assign_fallback_task(available_worker, config, plex, title_max_width):
                            attempted_workers = 0  # Reset counter on successful assignment
                            continue
                        # No fallback items - if main queue is also empty, break
                        if not media_queue:
                            break

                    # Assign from main queue (for GPU workers or when main queue has items)
                    assigned, queue_empty = self._assign_main_queue_task(available_worker, media_queue, config, plex, title_max_width)
                    if queue_empty:
                        # Queue is empty, stop trying to assign
                        break
                    if assigned:
                        # Successfully assigned, reset counter
                        attempted_workers = 0
                    else:
                        # Duplicate detected or other failure, increment counter
                        attempted_workers += 1

            # Dynamic queue refilling: fetch more items if queue is running low
            if fetch_more_items and len(media_queue) < len(self.workers):
                # Queue is running low, try to fetch more items
                try:
                    new_items = fetch_more_items()
                    if new_items:
                        media_queue.extend(new_items)
                        total_items += len(new_items)
                        logger.debug(f"Fetched {len(new_items)} more items from queue (total now: {total_items})")
                except Exception as e:
                    logger.error(f"Error fetching more items: {e}")

            # Check exit condition if stopped and idle
            if stop_condition and stop_condition() and not self.has_busy_workers():
                logger.info("Processing paused/stopped and all workers finished.")
                break

            # Check exit condition after trying to assign all tasks
            # Exit only if: main queue empty, all items processed, and fallback queue empty
            if not media_queue:
                # Re-check completion one more time (workers might have just finished)
                for worker in self.workers:
                    if worker.check_completion():
                        completed_tasks += 1
                        progress_manager.update_main_progress(completed_tasks, total_items)
                
                # Calculate actual completed count from worker stats (most reliable)
                actual_completed = sum(worker.completed for worker in self.workers)
                actual_failed = sum(worker.failed for worker in self.workers)
                actual_skipped = sum(worker.skipped for worker in self.workers)
                actual_processed = actual_completed + actual_failed + actual_skipped

                # Exit if all items processed (completed, failed, or skipped)
                if actual_processed >= total_items:
                    # Give threads time to finish and update is_busy flags
                    busy_retries = 0
                    max_busy_retries = 20  # Wait up to 20ms for threads to finish
                    while self.has_busy_workers() and busy_retries < max_busy_retries:
                        time.sleep(0.001)  # 1ms delay
                        for worker in self.workers:
                            worker.check_completion()
                        busy_retries += 1
                    
                    # After retries, check if we should exit
                    should_exit = (not self.has_busy_workers() or 
                                  (busy_retries >= max_busy_retries and actual_processed >= total_items))
                    
                    if should_exit:
                        if busy_retries >= max_busy_retries and actual_processed >= total_items:
                            # Log that we're exiting after waiting
                            logger.debug(f"All items processed ({actual_processed}/{total_items}), exiting after {busy_retries} retries")
                        
                        # Check fallback queue - if empty, we're done
                        if self._check_fallback_queue_empty():
                            break
            
            # Adaptive sleep to balance responsiveness and CPU usage
            if self.has_busy_workers():
                time.sleep(0.005)  # 5ms sleep for better responsiveness with multiple workers
            elif not media_queue:
                # No busy workers and no main queue items - give a tiny delay to ensure workers finished
                time.sleep(0.001)  # 1ms sleep when idle to let threads finish
        
        # Final statistics
        total_completed = sum(worker.completed for worker in self.workers)
        total_failed = sum(worker.failed for worker in self.workers)
        total_skipped = sum(worker.skipped for worker in self.workers)

        # Clean up worker progress
        progress_manager.cleanup_workers()

        # Build statistics message
        stats_parts = [f'{total_completed} successful']
        if total_failed > 0:
            stats_parts.append(f'{total_failed} failed')
        if total_skipped > 0:
            stats_parts.append(f'{total_skipped} skipped (not found)')

        logger.info(f'Processing complete: {", ".join(stats_parts)}')
    
    def _update_worker_progress(self, worker, progress_percent, current_duration, total_duration, speed=None,
                               remaining_time=None, frame=0, fps=0, q=0, size=0, time_str="00:00:00.00", bitrate=0, media_file=None, failed=False, error_message=None, avg_speed=None):
        """Update worker progress data from callback."""
        # Use thread-safe updates to prevent race conditions
        with self._progress_lock:
            worker.progress_percent = progress_percent
            worker.current_duration = current_duration
            worker.total_duration = total_duration
            if failed:
                worker.failed_state = True
                if error_message:
                    worker.error_message = error_message
            if speed:
                worker.speed = speed
            if avg_speed:
                worker.avg_speed = avg_speed
            if remaining_time is not None:
                worker.remaining_time = remaining_time

            # Store media file path if provided
            if media_file:
                worker.media_file = media_file

            # Store FFmpeg data for display
            worker.frame = frame
            worker.fps = fps
            worker.q = q
            worker.size = size
            worker.time_str = time_str
            worker.bitrate = bitrate
            
            # Log when FFmpeg actually starts processing (only once)
            if not worker.ffmpeg_started:
                display_path = worker.media_file if worker.media_file else worker.media_title
                if worker.worker_type == 'GPU':
                    logger.info(f"[GPU {worker.gpu_index}]: Started processing {display_path}")
                else:
                    logger.info(f"[CPU]: Started processing {display_path}")
            
            # Mark that FFmpeg has started outputting progress
            worker.ffmpeg_started = True
            
            # Emit periodic progress logs every 5 seconds
            current_time = time.time()
            if current_time - worker.last_verbose_log_time >= 5.0:
                worker.last_verbose_log_time = current_time
                speed_display = speed if speed else "0.0x"
                if worker.worker_type == 'GPU':
                    logger.info(f"[GPU {worker.gpu_index}]: {worker.media_title} - {progress_percent}% (speed={speed_display})")
                else:
                    logger.info(f"[CPU]: {worker.media_title} - {progress_percent}% (speed={speed_display})")
    
    def shutdown(self) -> None:
        """Shutdown all workers gracefully."""
        logger.debug("Shutting down worker pool...")
        for worker in self.workers:
            worker.shutdown()
        logger.debug("Worker pool shutdown complete")
