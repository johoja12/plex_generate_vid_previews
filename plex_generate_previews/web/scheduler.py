import threading
import time
import logging
import json
import sqlite3
from typing import List, Tuple, Optional
from itertools import chain
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlmodel import Session, select, col
from sqlalchemy import or_
from loguru import logger

from ..config import Config, load_config
from ..plex_client import plex_server, get_library_sections, get_media_parts_from_database, get_all_media_parts_batch, get_watch_history_from_database, retry_plex_call
from ..worker import WorkerPool
from ..gpu_detection import detect_all_gpus
from ..utils import setup_working_directory, sanitize_path
from .database import engine, get_session
from .models import MediaItem, PreviewStatus, MediaType, AppSettings
from .priority import (
    EpisodeRow,
    HubItem,
    PriorityInfo,
    WatchEvent,
    RECENT_WATCHED_SCORE,
    add_reason,
    build_priority_infos,
)
import os
from datetime import datetime, timezone, timedelta

class DbProgressManager:
    def __init__(self):
        # Track last update time per worker to throttle DB writes
        self.last_update = {}
        # Batch updates in memory before writing to DB
        self.pending_updates = {}  # {item_key: data}
        self.update_lock = threading.Lock()
        self.recorded_bytes = {}  # {item_key: last_recorded_bytes}
        self.last_batch_write = time.time()
        self.batch_write_interval = 5.0  # Write batches every 5 seconds (increased from 2)

    def init_workers(self, workers):
        pass

    def update_worker(self, worker_id, data: dict):
        item_key = data.get('item_key')
        if not item_key:
            logger.debug(f"DbProgressManager: Skipping update - no item_key in data")
            return

        progress = data.get('progress_percent', 0)
        is_failed = data.get('failed', False)
        processed_bytes = data.get('processed_bytes', 0)

        # Record data usage increments
        with self.update_lock:
            last_bytes = self.recorded_bytes.get(item_key, 0)
            if processed_bytes > last_bytes:
                delta = processed_bytes - last_bytes
                logger.debug(f"Usage delta for {item_key}: {delta} bytes (current: {processed_bytes}, last: {last_bytes})")
                scheduler._record_processed_data(item_key, delta)
                self.recorded_bytes[item_key] = processed_bytes

        # Debug log for failed items
        if is_failed:
            logger.debug(f"DbProgressManager: Received failed=True for item {item_key}, progress={progress}, error={data.get('error_message')}")

        # Critical updates that bypass batching and write immediately
        is_critical = progress == 0 or progress >= 100 or is_failed

        if is_critical:
            # Write immediately for critical updates (start, complete, fail)
            self._write_update_immediate(item_key, data)
            # Clear any pending batched updates for this item to prevent stale updates from overwriting terminal states
            with self.update_lock:
                if item_key in self.pending_updates:
                    del self.pending_updates[item_key]
                    logger.debug(f"Cleared pending batch updates for item {item_key} after critical update")
        else:
            # Batch non-critical updates (progress updates during processing)
            with self.update_lock:
                # Store/update pending data for this item
                self.pending_updates[item_key] = data

                # Check if it's time to flush the batch
                current_time = time.time()
                if current_time - self.last_batch_write >= self.batch_write_interval:
                    self._flush_batch_updates()
                    self.last_batch_write = current_time

    def _write_update_immediate(self, item_key, data):
        """Write a single critical update immediately with retry logic."""
        max_retries = 5
        retry_delay = 0.05  # Start with 50ms

        for attempt in range(max_retries):
            try:
                with Session(engine) as session:
                    item = session.get(MediaItem, int(item_key))
                    if not item:
                        logger.warning(f"DbProgressManager: Item {item_key} not found in database")
                        return

                    self._apply_update_to_item(item, data)
                    session.add(item)
                    session.commit()
                    return  # Success

            except Exception as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Gentler exponential backoff
                else:
                    if attempt == max_retries - 1:
                        logger.warning(f"⚠️  - Failed to update DB for {item_key} after {max_retries} attempts: {e}")
                    else:
                        logger.error(f"Failed to update DB for {item_key}: {e}")
                    return

    def _flush_batch_updates(self):
        """Flush all pending updates in a single transaction."""
        if not self.pending_updates:
            return

        # Copy and clear pending updates
        updates_to_write = dict(self.pending_updates)
        self.pending_updates.clear()

        max_retries = 5
        retry_delay = 0.05

        for attempt in range(max_retries):
            try:
                with Session(engine) as session:
                    # Process all updates in a single transaction
                    for item_key, data in updates_to_write.items():
                        item = session.get(MediaItem, int(item_key))
                        if item:
                            self._apply_update_to_item(item, data)
                            session.add(item)

                    session.commit()
                    logger.debug(f"Batch wrote {len(updates_to_write)} progress updates to DB")
                    return  # Success

            except Exception as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    if attempt == max_retries - 1:
                        logger.warning(f"⚠️  - Failed to batch write {len(updates_to_write)} updates after {max_retries} attempts: {e}")
                    else:
                        logger.error(f"Failed to batch write updates: {e}")
                    return

    def _apply_update_to_item(self, item, data):
        """Apply update data to a MediaItem object."""
        progress = data.get('progress_percent', 0)
        error_message = data.get('error_message')
        media_file = data.get('media_file')
        bundle_hash = data.get('bundle_hash')
        avg_speed = data.get('avg_speed')

        if data.get('failed'):
            # Task failed - check if it's slow processing
            if error_message and "too slow" in error_message.lower():
                item.status = PreviewStatus.SLOW_FAILED
                logger.info(f"Marked item {item.id} ({item.title}) as SLOW_FAILED: {error_message}")
            else:
                item.status = PreviewStatus.FAILED
                logger.info(f"Marked item {item.id} ({item.title}) as FAILED: {error_message}")
            item.progress = int(progress) if progress > 0 else item.progress
            item.error_message = error_message
            if avg_speed:
                item.avg_speed = avg_speed
            if bundle_hash:
                item.current_processing_bundle_hash = bundle_hash

        elif progress >= 100:
            # Completed
            item.status = PreviewStatus.COMPLETED
            item.progress = 100
            item.error_message = None
            if avg_speed:
                item.avg_speed = avg_speed
            if bundle_hash:
                item.current_processing_bundle_hash = bundle_hash
            
            # Set BIF path if we have a bundle hash (using primary hash if available, or the one just processed)
            # Note: For multi-part items, this might just set the path to the last processed one, 
            # which is fine as MediaItem.bif_path is mostly for legacy/single-part compatibility.
            target_hash = bundle_hash if bundle_hash else item.bundle_hash
            if target_hash:
                from ..utils import sanitize_path
                bundle_file = sanitize_path(f'{target_hash[0]}/{target_hash[1::1]}.bundle')
                bundle_path = sanitize_path(os.path.join(
                    scheduler.config.plex_config_folder if scheduler.config else '/config/plex',
                    'Media', 'localhost', bundle_file
                ))
                bif_path = sanitize_path(os.path.join(bundle_path, 'Contents', 'Indexes', 'index-sd.bif'))
                item.bif_path = bif_path
            logger.info(f"Marked item {item.id} ({item.title}) as COMPLETED")

        elif progress > 0:
            # Processing - but don't overwrite terminal states (FAILED, COMPLETED, SLOW_FAILED)
            # This prevents stale batched updates from overwriting critical state changes
            if item.status not in [PreviewStatus.FAILED, PreviewStatus.SLOW_FAILED, PreviewStatus.COMPLETED]:
                item.status = PreviewStatus.PROCESSING
            item.progress = int(progress)
            if avg_speed:
                item.avg_speed = avg_speed
            if media_file:
                item.file_path = media_file
            if bundle_hash:
                item.current_processing_bundle_hash = bundle_hash
        
        elif progress == 0 and not data.get('failed'):
            # Started processing a new part (0% progress)
            # Don't reset status if it was already processing (e.g. part 2 starting after part 1)
            if item.status == PreviewStatus.COMPLETED:
                item.status = PreviewStatus.PROCESSING
            elif item.status == PreviewStatus.MISSING or item.status == PreviewStatus.QUEUED:
                item.status = PreviewStatus.PROCESSING
            
            item.progress = 0
            if bundle_hash:
                item.current_processing_bundle_hash = bundle_hash

    def update_main_progress(self, completed, total):
        pass

    def cleanup_workers(self):
        """Flush any remaining pending updates before shutdown."""
        with self.update_lock:
            if self.pending_updates:
                logger.debug(f"Flushing {len(self.pending_updates)} pending updates on cleanup")
                self._flush_batch_updates()

class Scheduler:
    def __init__(self):
        self.config: Optional[Config] = None
        self.running = False
        self.thread = None
        self.sync_thread = None
        self.worker_pool = None
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.sync_wake_event = threading.Event()  # Separate event for sync thread
        self.force_sync = False
        self.paused = False
        self.last_sync_time: Optional[datetime] = None # Will be loaded from DB
        self.sync_in_progress = False  # Track if sync is currently running
        self._last_rate_limit_check = 0
        self._rate_limited = False
        self._rate_limit_message = ""
        self._rate_limit_exempt_paths_cache = []  # Cache for exempt paths
        self._rate_limit_exempt_paths_cache_time = 0  # Timestamp of last cache update
        self._priority_refresh_lock = threading.Lock()

    @staticmethod
    def _priority_order_by():
        return (
            MediaItem.priority_score.desc(),
            MediaItem.updated_at.desc(),
            MediaItem.queue_order.asc(),
        )

    def _apply_priority_info_to_item(self, item: MediaItem, priority_info: Optional[PriorityInfo]) -> bool:
        calculated_at = datetime.utcnow()
        if priority_info and priority_info.score > 0:
            priority_reasons = json.dumps(priority_info.reasons, separators=(",", ":"))
            changed = (
                item.priority_score != priority_info.score
                or item.priority_reasons != priority_reasons
                or not item.is_priority
            )
            item.is_priority = True
            item.priority_score = priority_info.score
            item.priority_reasons = priority_reasons
            item.priority_last_calculated_at = calculated_at
            return changed

        changed = (
            item.is_priority
            or item.priority_score != 0
            or item.priority_reasons is not None
            or item.priority_last_calculated_at is not None
        )
        item.is_priority = False
        item.priority_score = 0
        item.priority_reasons = None
        item.priority_last_calculated_at = None
        return changed

    def _clear_completed_priority_metadata(self, item: MediaItem) -> bool:
        if item.status != PreviewStatus.COMPLETED:
            return False
        return self._apply_priority_info_to_item(item, None)

    def _priority_info_for_key(self, priority_items, item_key: int) -> Optional[PriorityInfo]:
        if hasattr(priority_items, "get"):
            return priority_items.get(item_key)
        if item_key in priority_items:
            info = PriorityInfo()
            add_reason(info, "legacy_priority", RECENT_WATCHED_SCORE)
            return info
        return None

    def trigger_sync(self):
        self.force_sync = True
        logger.info("Manual sync requested")
        self.sync_wake_event.set()  # Wake sync thread for immediate sync

    def refresh_priority_metadata(self, plex=None) -> dict[str, int]:
        """Refresh priority score/reason metadata for existing queued work only."""
        if not self.config and plex is None:
            logger.debug("Skipping priority metadata refresh: no config loaded")
            return {"candidates": 0, "scored": 0, "updated": 0}

        if not self._priority_refresh_lock.acquire(blocking=False):
            logger.debug("Priority metadata refresh already running")
            return {"candidates": 0, "scored": 0, "updated": 0}

        try:
            candidate_statuses = [
                PreviewStatus.MISSING,
                PreviewStatus.QUEUED,
                PreviewStatus.PROCESSING,
            ]
            with Session(engine) as session:
                candidate_ids = set(
                    session.exec(
                        select(MediaItem.id).where(col(MediaItem.status).in_(candidate_statuses))
                    ).all()
                )

            if plex is None:
                plex = plex_server(self.config)

            priority_infos = self.detect_priority_items(plex, missing_rating_keys=candidate_ids)

            updated_count = 0
            with Session(engine) as session:
                rows = session.exec(
                    select(MediaItem).where(
                        or_(
                            col(MediaItem.status).in_(candidate_statuses),
                            MediaItem.is_priority == True,
                        )
                    )
                ).all()

                for item in rows:
                    if item.status in candidate_statuses:
                        changed = self._apply_priority_info_to_item(
                            item,
                            priority_infos.get(item.id),
                        )
                    else:
                        changed = self._clear_completed_priority_metadata(item)

                    if changed:
                        session.add(item)
                        updated_count += 1

                session.commit()

            summary = {
                "candidates": len(candidate_ids),
                "scored": len(priority_infos),
                "updated": updated_count,
            }
            logger.info(
                "Priority metadata refresh complete: "
                f"{summary['scored']}/{summary['candidates']} candidates scored, "
                f"{summary['updated']} rows updated"
            )
            return summary

        except Exception as e:
            logger.error(f"Priority metadata refresh failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"candidates": 0, "scored": 0, "updated": 0}
        finally:
            self._priority_refresh_lock.release()

    def validate_mount_paths(self) -> tuple[bool, list[str]]:
        """
        Validate that configured mount check paths exist and are accessible.

        Returns:
            tuple: (all_valid, list_of_missing_paths)
        """
        try:
            with Session(engine) as session:
                settings = session.get(AppSettings, 1)

                # If mount check is disabled, always return valid
                if not settings or not settings.mount_check_enabled:
                    logger.debug("Mount validation disabled, skipping check")
                    return True, []

                # If no paths configured, check common media locations
                check_paths = []
                if settings.mount_check_paths:
                    check_paths = [p.strip() for p in settings.mount_check_paths.split(',') if p.strip()]

                if not check_paths:
                    logger.debug("No mount check paths configured, validation passes")
                    return True, []

                missing_paths = []
                for path in check_paths:
                    if not os.path.exists(path):
                        missing_paths.append(path)
                        logger.warning(f"Mount check failed: Path does not exist: {path}")
                    else:
                        # Try to check if it's actually accessible (test read)
                        try:
                            if os.path.isfile(path):
                                # For files, try to stat them
                                os.stat(path)
                            elif os.path.isdir(path):
                                # For directories, try to list them
                                os.listdir(path)
                            logger.debug(f"Mount check passed: {path}")
                        except (OSError, PermissionError) as e:
                            missing_paths.append(path)
                            logger.warning(f"Mount check failed: Path exists but not accessible: {path} ({e})")

                if missing_paths:
                    return False, missing_paths
                else:
                    return True, []

        except Exception as e:
            logger.error(f"Mount validation error: {e}")
            # On error, assume mounts are OK to avoid false positives
            return True, []

    def pause(self):
        self.paused = True
        logger.info("Queue paused")
        # Persist to database
        try:
            with Session(engine) as session:
                settings = session.get(AppSettings, 1)
                if not settings:
                    settings = AppSettings(id=1)
                settings.queue_paused = True
                session.add(settings)
                session.commit()
        except Exception as e:
            logger.error(f"Failed to persist pause state to database: {e}")

    def resume(self):
        self.paused = False
        logger.info("Queue resumed")
        # Persist to database
        try:
            with Session(engine) as session:
                settings = session.get(AppSettings, 1)
                if not settings:
                    settings = AppSettings(id=1)
                settings.queue_paused = False
                session.add(settings)
                session.commit()
        except Exception as e:
            logger.error(f"Failed to persist resume state to database: {e}")

    def _check_rate_limit(self) -> bool:
        """
        Check if we've exceeded the hourly data limit.
        Returns True if rate limited.
        """
        try:
            with Session(engine) as session:
                settings = session.get(AppSettings, 1)
                if not settings or not settings.data_limit_gb_per_hour or settings.data_limit_gb_per_hour <= 0:
                    self._rate_limited = False
                    # Clear debt if limit is disabled
                    if settings and settings.total_bytes_processed_hour > 0:
                        settings.total_bytes_processed_hour = 0
                        session.add(settings)
                        session.commit()
                    return False

                now = datetime.utcnow()
                limit_bytes = int(settings.data_limit_gb_per_hour * 1024 * 1024 * 1024)
                
                # Check if an hour has passed since the current tracking period started
                seconds_passed = (now - settings.hour_start_time).total_seconds()
                if seconds_passed >= 3600:
                    hours_passed = int(seconds_passed // 3600)
                    # "Borrowing" implementation: Subtract Quota for passed hours
                    # This carries over any excess (debt) to the next hour
                    settings.total_bytes_processed_hour = max(0, settings.total_bytes_processed_hour - (hours_passed * limit_bytes))
                    settings.hour_start_time = settings.hour_start_time + timedelta(hours=hours_passed)
                    
                    session.add(settings)
                    session.commit()
                    logger.info(f"Rate limit period reset for {hours_passed} hour(s). Current debt: {settings.total_bytes_processed_hour / 1024 / 1024 / 1024:.2f} GB")
                    
                    # Re-evaluate after reset
                    if settings.total_bytes_processed_hour < limit_bytes:
                        self._rate_limited = False
                        return False

                if settings.total_bytes_processed_hour >= limit_bytes:
                    if not self._rate_limited:
                        wait_seconds = int(3600 - (seconds_passed % 3600))
                        self._rate_limit_message = f"Hourly rate limit reached ({settings.data_limit_gb_per_hour} GB). Debt: {settings.total_bytes_processed_hour / 1024 / 1024 / 1024:.2f} GB. Waiting {wait_seconds}s for next hour."
                        logger.warning(self._rate_limit_message)
                        self._rate_limited = True
                    return True

                self._rate_limited = False
                return False
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False

    def _get_rate_limit_exempt_paths(self) -> list:
        """
        Get the list of paths exempt from rate limiting, with caching.
        Returns a list of path prefixes that are exempt.
        """
        # Cache for 60 seconds to avoid frequent DB queries
        now = time.time()
        if now - self._rate_limit_exempt_paths_cache_time < 60 and self._rate_limit_exempt_paths_cache is not None:
            return self._rate_limit_exempt_paths_cache

        try:
            with Session(engine) as session:
                settings = session.get(AppSettings, 1)
                if settings and settings.rate_limit_exempt_paths:
                    # Parse comma-separated paths, strip whitespace
                    paths = [p.strip() for p in settings.rate_limit_exempt_paths.split(',') if p.strip()]
                    self._rate_limit_exempt_paths_cache = paths
                else:
                    self._rate_limit_exempt_paths_cache = []
                self._rate_limit_exempt_paths_cache_time = now
                return self._rate_limit_exempt_paths_cache
        except Exception as e:
            logger.error(f"Error getting rate limit exempt paths: {e}")
            return []

    def _is_path_rate_limit_exempt(self, file_path: str) -> bool:
        """
        Check if a file path (or its symlink target) is exempt from rate limiting.

        This checks both the original path AND the fully resolved real path,
        which handles cases where:
        - The file itself is a symlink
        - Any parent directory in the path is a symlink

        Args:
            file_path: The file path to check

        Returns:
            True if the path is exempt from rate limiting, False otherwise
        """
        if not file_path:
            return False

        exempt_paths = self._get_rate_limit_exempt_paths()
        if not exempt_paths:
            return False  # No exempt paths configured

        # Check the file path itself
        for exempt_path in exempt_paths:
            if file_path.startswith(exempt_path):
                logger.debug(f"Path {file_path} is rate limit exempt (matches {exempt_path})")
                return True

        # Always resolve the full path to handle symlinks (both file and parent directories)
        try:
            real_path = os.path.realpath(file_path)
            # Only check if the resolved path is different from the original
            if real_path != file_path:
                for exempt_path in exempt_paths:
                    if real_path.startswith(exempt_path):
                        logger.debug(f"Resolved path {real_path} (from {file_path}) is rate limit exempt (matches {exempt_path})")
                        return True
        except (OSError, IOError) as e:
            logger.debug(f"Could not resolve path {file_path}: {e}")

        return False

    def _is_item_rate_limit_exempt(self, item: MediaItem) -> bool:
        """
        Check if a media item is exempt from rate limiting based on its file paths.

        Args:
            item: MediaItem to check

        Returns:
            True if the item is exempt from rate limiting
        """
        import json

        # Check all media parts for the item
        if item.media_parts_info:
            try:
                parts = json.loads(item.media_parts_info)
                for part in parts:
                    file_path = part.get('file_path')
                    if file_path and self._is_path_rate_limit_exempt(file_path):
                        return True
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Could not parse media_parts_info for item {item.id}: {e}")

        # Fallback to file_path field
        if item.file_path and self._is_path_rate_limit_exempt(item.file_path):
            return True

        return False

    def _record_processed_data(self, item_id: str, bytes_processed: int):
        """
        Update the hourly processed bytes count.
        Skips recording for rate-limit exempt items.
        """
        if bytes_processed <= 0:
            return

        # Check if this item is rate-limit exempt
        try:
            with Session(engine) as session:
                item = session.get(MediaItem, int(item_id))
                if item and self._is_item_rate_limit_exempt(item):
                    logger.debug(f"Skipping data recording for rate-limit exempt item {item_id} ({item.title})")
                    return
        except Exception as e:
            logger.debug(f"Could not check rate limit exemption for item {item_id}: {e}")

        logger.debug(f"Recording {bytes_processed} bytes for item {item_id}")
        try:
            with Session(engine) as session:
                # Use a raw update to be more atomic and handle concurrency better
                from sqlalchemy import update
                session.execute(
                    update(AppSettings)
                    .where(AppSettings.id == 1)
                    .values(total_bytes_processed_hour=AppSettings.total_bytes_processed_hour + bytes_processed)
                )
                session.commit()
        except Exception as e:
            logger.error(f"Failed to record processed data for item {item_id}: {e}")

    def _get_bif_path(self, bundle_hash: str) -> str:
        """Construct BIF file path from bundle hash"""
        if not bundle_hash or len(bundle_hash) < 2:
            return None
        from ..utils import sanitize_path
        bundle_file = sanitize_path(f'{bundle_hash[0]}/{bundle_hash}.bundle')
        bundle_path = sanitize_path(os.path.join(
            self.config.plex_config_folder,
            'Media', 'localhost', bundle_file
        ))
        bif_path = sanitize_path(os.path.join(bundle_path, 'Contents', 'Indexes', 'index-sd.bif'))
        return bif_path

    def start(self):
        if self.running:
            return
        
        # Initialize running state
        self.running = True
        self.stop_event.clear()
        self.wake_event.clear()
        
        # Try to load config initially
        self._try_load_config()
        
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        # Start background sync thread
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()

        if self.config:
            threading.Thread(target=self.refresh_priority_metadata, daemon=True).start()

        logger.info("Scheduler started")

    def stop(self):
        self.running = False
        self.stop_event.set()
        self.wake_event.set()
        if self.worker_pool:
            self.worker_pool.shutdown()
        if self.thread:
            self.thread.join()
        if self.sync_thread:
            self.sync_thread.join()
        logger.info("Scheduler stopped")
    
    def _get_selected_gpus(self):
        # Replicated logic from cli.py to avoid sys.exit and dependency
        if self.config.gpu_threads == 0:
            return []
            
        detected_gpus = detect_all_gpus()
        if not detected_gpus:
            logger.warning("No GPUs detected, but GPU threads requested. Fallback logic might apply.")
            return []

        if self.config.gpu_selection.lower() == 'all':
            return detected_gpus
        
        try:
            indices = [int(x.strip()) for x in self.config.gpu_selection.split(',') if x.strip()]
            selected = []
            for idx in indices:
                if 0 <= idx < len(detected_gpus):
                    selected.append(detected_gpus[idx])
            return selected
        except:
            return detected_gpus # Fallback
    
    def _try_load_config(self):
        """Attempt to load configuration from Env or DB."""
        try:
            # 1. Try Env/CLI config
            # We need to catch SystemExit because load_config might exit on failure if we aren't careful
            # But currently load_config returns None on failure in most cases (except argument parsing help)
            # However, we should be careful.
            # Actually, let's try to load from DB first to overlay? 
            # No, Env should take precedence usually, but for "Setup" flow, DB is key.
            
            # Let's check DB first.
            from .models import AppSettings
            db_config = None
            try:
                with Session(engine) as session:
                    db_config = session.get(AppSettings, 1)
            except Exception as e:
                logger.warning(f"Could not load settings from DB: {e}")

            # Load Env Config
            env_config = load_config(None, print_help=False) # This might return None if invalid
            
            if env_config:
                self.config = env_config
                # If DB has settings, maybe we should use them if Env is missing specific ones?
                # For now, let's assume if Env is valid, we use it. 
                # BUT, if we want to allow web-based setup to override, we need to merge.
                # Simplest for now: If Env config is valid, use it. 
                # If Env config is INVALID (None), try to build from DB.
            elif db_config and db_config.plex_url and db_config.plex_token:
                # Construct Config from DB
                # We need to fill in defaults for other fields
                self.config = Config(
                    plex_url=db_config.plex_url,
                    plex_token=db_config.plex_token,
                    plex_timeout=60,
                    plex_libraries=[], # Default to all? or need to store in DB
                    plex_config_folder= "/config/plex", # Default or DB?
                    plex_local_videos_path_mapping="",
                    plex_videos_path_mapping="",
                    plex_path_mappings=[],
                    plex_bif_frame_interval=5,
                    thumbnail_quality=4,
                    regenerate_thumbnails=False,
                    sort_by="newest",
                    gpu_threads=db_config.gpu_threads,
                    cpu_threads=db_config.cpu_threads,
                    gpu_selection="all",
                    tmp_folder="/tmp",
                    tmp_folder_created_by_us=False,
                    ffmpeg_path="ffmpeg", # Assumption
                    log_level="INFO",
                    scheduler_loop_interval=db_config.scheduler_loop_interval
                )
                # Set env var for plexapi
                import os
                os.environ["PLEXAPI_TIMEOUT"] = str(self.config.plex_timeout)
                logger.info("Loaded configuration from Database")
            else:
                logger.warning("No valid configuration found (Env or DB). Scheduler waiting for setup.")
                self.config = None

            if self.config:
                # Update last_sync_time from DB if available
                if db_config and db_config.last_sync_time:
                    self.last_sync_time = db_config.last_sync_time
                else:
                    self.last_sync_time = datetime.fromtimestamp(0) # Epoch start

                # Load paused state from DB if available
                if db_config and hasattr(db_config, 'queue_paused'):
                    self.paused = db_config.queue_paused
                    logger.info(f"Loaded queue paused state from database: {self.paused}")
                else:
                    self.paused = False

                # Setup working directory
                try:
                    self.config.working_tmp_folder = setup_working_directory(self.config.tmp_folder)
                    logger.info(f"Working directory set to: {self.config.working_tmp_folder}")
                except Exception as e:
                    logger.error(f"Failed to create working directory: {e}")
                    self.config = None
                    return

                # Initialize WorkerPool if we have config
                if not self.worker_pool:
                    selected_gpus = self._get_selected_gpus()
                    self.worker_pool = WorkerPool(
                        gpu_workers=self.config.gpu_threads,
                        cpu_workers=self.config.cpu_threads,
                        selected_gpus=selected_gpus
                    )

                    # Cleanup stale items after worker pool is initialized
                    # This ensures items stuck in PROCESSING/QUEUED from previous sessions get reset
                    logger.info("Running startup cleanup of stale items...")
                    self.cleanup_stale_items()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = None

    def _sync_loop(self):
        """Background thread for library sync operations to avoid blocking queue processing."""
        while not self.stop_event.is_set():
            try:
                if not self.config:
                    time.sleep(10)
                    continue

                # Get sync interval from settings (default: 6 hours = 21600 seconds)
                sync_interval = 21600  # Default to 6 hours
                try:
                    with Session(engine) as session:
                        settings = session.get(AppSettings, 1)
                        if settings and settings.sync_interval:
                            sync_interval = settings.sync_interval
                except Exception as e:
                    logger.debug(f"Could not load sync_interval from settings, using default: {e}")

                # Check if sync is needed (based on sync_interval or when forced)
                # Use datetime subtraction to avoid timezone issues with .timestamp()
                # (naive UTC datetimes compared with naive UTC datetimes)
                if self.last_sync_time:
                    seconds_since_sync = (datetime.utcnow() - self.last_sync_time).total_seconds()
                    sync_needed = self.force_sync or seconds_since_sync > sync_interval
                else:
                    sync_needed = True

                if sync_needed:
                    logger.info("Background sync: Starting library sync...")
                    self.refresh_priority_metadata()
                    self.sync_library()
                    # last_sync_time is already updated inside sync_library()
                    self.force_sync = False
                    self.sync_wake_event.clear()  # Clear the wake event after sync
                    logger.info("Background sync: Library sync complete")

                # Wait 60 seconds or until wake event is set (for manual sync trigger)
                # Using sync_wake_event.wait with timeout allows immediate response to manual sync
                self.sync_wake_event.wait(60)

            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                time.sleep(60)

    def _run_loop(self):
        while not self.stop_event.is_set(): # Main loop condition for stopping
            try:
                if not self.config:
                    # Retry loading config (maybe user completed setup)
                    self._try_load_config()
                    if not self.config:
                        time.sleep(5)
                        continue

                self.process_queue()

                # Clear wake event before waiting (if it was set)
                self.wake_event.clear()

                # Wait for next interval or wake signal (trigger_sync or stop)
                self.wake_event.wait(self.config.scheduler_loop_interval)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(10) # Short sleep on error
    def _get_bif_path(self, bundle_hash):
        if not bundle_hash or len(bundle_hash) < 2:
            return None
        bundle_file = sanitize_path(f'{bundle_hash[0]}/{bundle_hash[1::1]}.bundle')
        bundle_path = sanitize_path(os.path.join(self.config.plex_config_folder, 'Media', 'localhost', bundle_file))
        return sanitize_path(os.path.join(bundle_path, 'Contents', 'Indexes', 'index-sd.bif'))

    def _scan_filesystem_for_bifs(self):
        """
        Efficiently scan the filesystem for existing BIF files.

        Returns:
            dict: Mapping of bundle_hash -> bif_path for all found BIF files
        """
        import glob as glob_module
        import re

        if not self.config:
            logger.error("Cannot scan: scheduler not configured")
            return {}

        media_localhost_path = os.path.join(self.config.plex_config_folder, 'Media', 'localhost')

        if not os.path.exists(media_localhost_path):
            logger.warning(f"Media path does not exist: {media_localhost_path}")
            return {}

        logger.debug(f"Scanning filesystem for existing BIF files in {media_localhost_path}...")
        start_time = time.time()

        # Pattern to match BIF files: {plex_config}/Media/localhost/*/**.bundle/Contents/Indexes/index-sd.bif
        bif_pattern = os.path.join(media_localhost_path, '*', '*.bundle', 'Contents', 'Indexes', 'index-sd.bif')
        logger.debug(f"Glob pattern: {bif_pattern}")

        bif_files = glob_module.glob(bif_pattern)
        logger.debug(f"Found {len(bif_files)} BIF files in {time.time() - start_time:.2f} seconds")

        if len(bif_files) > 0:
            logger.debug(f"Sample found BIF paths: {bif_files[:3]}")

        # Extract bundle hash from each path
        # Path format: .../Media/localhost/{first_char}/{bundle_hash}.bundle/Contents/Indexes/index-sd.bif
        # Example: .../Media/localhost/9/95c7e436036530c98ed02aef85b421100eb8625.bundle/Contents/Indexes/index-sd.bif
        bundle_hash_map = {}
        # Make regex case-insensitive and handle various path separators if needed
        pattern = re.compile(r'[/\\]([a-f0-9])[/\\]([a-f0-9]+)\.bundle[/\\]Contents[/\\]Indexes[/\\]index-sd\.bif$', re.IGNORECASE)
        
        failed_parse_count = 0
        for bif_path in bif_files:
            match = pattern.search(bif_path)
            if match:
                first_char = match.group(1).lower()
                bundle_hash = match.group(2).lower()

                full_bundle_hash = first_char + bundle_hash
                bundle_hash_map[full_bundle_hash] = bif_path
            else:
                failed_parse_count += 1
                if failed_parse_count <= 5:
                    logger.debug(f"Could not extract bundle hash from path (regex mismatch): {bif_path}")

        logger.debug(f"Extracted {len(bundle_hash_map)} valid bundle hashes from BIF files")
        if failed_parse_count > 0:
            logger.debug(f"Failed to parse hashes for {failed_parse_count} BIF files. Check debug logs for samples.")
            
        return bundle_hash_map

    def discover_existing_bifs(self, confirm_threshold: bool = False):
        """Efficiently scan filesystem for BIF files and mark corresponding MISSING items as COMPLETED.

        This function scans the filesystem once for all BIF files, then correlates them back
        to media items in the database. Much more efficient than checking the filesystem for
        each individual item.

        Args:
            confirm_threshold: If True, bypass the 50% threshold check (user confirmed)
        """
        logger.debug("Discovering existing BIF files for MISSING items...")
        if not self.config:
            logger.error("Cannot discover: scheduler not configured")
            return {"scanned": 0, "found": 0, "error": "Scheduler not configured"}

        # Log config details for debugging
        logger.debug(f"Plex config folder: {self.config.plex_config_folder}")

        try:
            # Step 1: Scan filesystem for all BIF files (single pass)
            bundle_hash_map = self._scan_filesystem_for_bifs()

            if not bundle_hash_map:
                logger.info("No BIF files found in filesystem")
                return {"scanned": 0, "found": 0}

            with Session(engine) as session:
                # Step 2: Get all MISSING items
                statement = select(MediaItem).where(MediaItem.status == PreviewStatus.MISSING)
                missing_items = session.exec(statement).all()

                total_items = len(missing_items)
                logger.debug(f"Found {total_items} MISSING items in database")

                if total_items == 0:
                    return {"scanned": 0, "found": 0}

                # Step 3: Find MISSING items that have BIF files
                found_items = []
                scanned_count = 0
                plex = None

                for item in missing_items:
                    scanned_count += 1

                    # If item has no bundle_hash, try to get it from Plex database
                    if not item.bundle_hash:
                        logger.debug(f"Item {item.id} ({item.title}) has no bundle_hash, querying Plex database")
                        try:
                            import sqlite3
                            db_path = os.path.join(self.config.plex_config_folder, 'Plug-in Support', 'Databases',
                                                   'com.plexapp.plugins.library.db')
                            if os.path.exists(db_path):
                                conn = sqlite3.connect(db_path)
                                cursor = conn.cursor()
                                # Query media_parts.hash via JOIN (bundle hash is in media_parts, not metadata_items)
                                query = """
                                    SELECT media_parts.hash
                                    FROM media_parts
                                    JOIN media_items ON media_parts.media_item_id = media_items.id
                                    WHERE media_items.metadata_item_id = ?
                                    LIMIT 1
                                """
                                cursor.execute(query, (item.id,))
                                result = cursor.fetchone()
                                conn.close()

                                if result and result[0]:
                                    item.bundle_hash = result[0]
                                    # Update the item in the session
                                    session.add(item)
                                else:
                                    logger.debug(f"No hash found in Plex database for item {item.id}")
                            else:
                                logger.warning(f"Plex database not found at: {db_path}")
                        except Exception as e:
                            logger.warning(f"Failed to query Plex database for hash for item {item.id}: {e}")

                    # Now check if we have a matching BIF file
                    if item.bundle_hash and item.bundle_hash in bundle_hash_map:
                        bif_path = bundle_hash_map[item.bundle_hash]
                        found_items.append((item, bif_path))
                        # Log first few discoveries
                        if len(found_items) <= 5:
                            logger.debug(f"Found existing BIF for item {item.id} ({item.title}): {bif_path}")
                    elif not item.bundle_hash:
                        # Still no hash after database lookup - trigger Plex analysis as last resort
                        logger.debug(f"Item {item.id} ({item.title}) still has no bundle_hash after database lookup")
                        try:
                            if plex is None:
                                plex = plex_server(self.config)

                            logger.debug(f"Triggering analysis for item {item.id} ({item.title}) due to missing bundle hash")
                            plex_item = plex.fetchItem(int(item.id))
                            plex_item.analyze()
                        except Exception as e:
                            logger.warning(f"Failed to trigger analysis for item {item.id}: {e}")

                found_count = len(found_items)

                # Calculate percentage that would be moved
                percentage_found = (found_count / total_items) * 100 if total_items > 0 else 0

                # Failsafe: if more than 50% would be moved, require confirmation
                if percentage_found > 50 and not confirm_threshold:
                    logger.warning(f"Discovery would move {found_count}/{total_items} ({percentage_found:.1f}%) items to COMPLETED. Requires confirmation.")
                    # Don't commit changes, return warning
                    return {
                        "scanned": scanned_count,
                        "found": 0,
                        "requires_confirmation": True,
                        "percentage": percentage_found,
                        "would_move": found_count,
                        "sample_paths": [path for _, path in found_items[:3]]  # Show sample paths for verification
                    }

                # Step 4: Actually mark items as COMPLETED
                for item, bif_path in found_items:
                    logger.debug(f"Marking item {item.id} ({item.title}) as COMPLETED (existing BIF found)")
                    item.status = PreviewStatus.COMPLETED
                    item.progress = 100
                    session.add(item)

                session.commit()

            logger.info(f"Discovery complete: {scanned_count} items scanned, {found_count} marked as COMPLETED")
            return {"scanned": scanned_count, "found": found_count}

        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"scanned": 0, "found": 0, "error": str(e)}

    def cleanup_orphaned_bundles(self, dry_run: bool = True):
        """
        Scan for orphaned bundle directories (not in Plex DB) and optionally delete them.

        Args:
            dry_run: If True, only report what would be deleted without actually deleting

        Returns:
            dict: Results with found/deleted counts and paths
        """
        logger.info(f"Scanning for orphaned bundle directories (dry_run={dry_run})...")

        if not self.config:
            logger.error("Cannot cleanup: scheduler not configured")
            return {"error": "Scheduler not configured"}

        try:
            import sqlite3
            import shutil

            # Get Plex database path
            db_path = os.path.join(self.config.plex_config_folder, 'Plug-in Support', 'Databases', 'com.plexapp.plugins.library.db')

            if not os.path.exists(db_path):
                logger.error(f"Plex database not found at: {db_path}")
                return {"error": "Plex database not found"}

            # Get all bundle hashes from Plex database (from media_parts, not metadata_items)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT hash FROM media_parts WHERE hash IS NOT NULL")
            db_hashes = set(row[0] for row in cursor.fetchall())
            conn.close()

            logger.info(f"Found {len(db_hashes)} bundle hashes in Plex database")

            # Scan filesystem for bundle directories
            media_localhost_path = os.path.join(self.config.plex_config_folder, 'Media', 'localhost')

            if not os.path.exists(media_localhost_path):
                logger.warning(f"Media path does not exist: {media_localhost_path}")
                return {"error": "Media path not found"}

            orphaned_bundles = []
            total_size = 0

            # Iterate through all bundle directories
            for first_char_dir in os.listdir(media_localhost_path):
                first_char_path = os.path.join(media_localhost_path, first_char_dir)

                if not os.path.isdir(first_char_path):
                    continue

                for bundle_dir in os.listdir(first_char_path):
                    if not bundle_dir.endswith('.bundle'):
                        continue

                    bundle_path = os.path.join(first_char_path, bundle_dir)

                    # Extract hash from directory name
                    # Format: {hash}.bundle, and full hash is first_char + hash
                    bundle_hash_part = bundle_dir.replace('.bundle', '')
                    full_hash = first_char_dir + bundle_hash_part

                    # Check if this hash exists in Plex database
                    if full_hash not in db_hashes:
                        # Calculate size
                        try:
                            dir_size = sum(
                                os.path.getsize(os.path.join(dirpath, filename))
                                for dirpath, dirnames, filenames in os.walk(bundle_path)
                                for filename in filenames
                            )
                            total_size += dir_size
                            orphaned_bundles.append({
                                'path': bundle_path,
                                'hash': full_hash,
                                'size': dir_size
                            })
                        except Exception as e:
                            logger.warning(f"Could not calculate size for {bundle_path}: {e}")

            logger.info(f"Found {len(orphaned_bundles)} orphaned bundle directories (total size: {total_size / 1024 / 1024:.2f} MB)")

            deleted_count = 0
            deleted_size = 0
            errors = []

            if not dry_run:
                # Actually delete the orphaned bundles
                for bundle in orphaned_bundles:
                    try:
                        logger.info(f"Deleting orphaned bundle: {bundle['path']}")
                        shutil.rmtree(bundle['path'])
                        deleted_count += 1
                        deleted_size += bundle['size']
                    except Exception as e:
                        error_msg = f"Failed to delete {bundle['path']}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)

                logger.info(f"Deleted {deleted_count} orphaned bundles (freed {deleted_size / 1024 / 1024:.2f} MB)")
            else:
                logger.info("Dry run - no bundles deleted")

            return {
                "found": len(orphaned_bundles),
                "deleted": deleted_count,
                "total_size": total_size,
                "deleted_size": deleted_size,
                "dry_run": dry_run,
                "errors": errors,
                "sample_paths": [b['path'] for b in orphaned_bundles[:5]]
            }

        except Exception as e:
            logger.error(f"Orphaned bundle cleanup failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def verify_completed_items(self, confirm_threshold: bool = False):
        """Verify that COMPLETED items actually have BIF files. Move to MISSING if not.

        Args:
            confirm_threshold: If True, bypass the 50% threshold check (user confirmed)
        """
        logger.info("Verifying completed items...")
        if not self.config:
            logger.error("Cannot verify: scheduler not configured")
            return {"verified": 0, "moved_to_missing": 0, "error": "Scheduler not configured"}

        # Log config details for debugging
        logger.info(f"Plex config folder: {self.config.plex_config_folder}")

        verified_count = 0
        moved_count = 0
        missing_items = []  # Store items that would be moved

        try:
            with Session(engine) as session:
                # Get all COMPLETED items
                statement = select(MediaItem).where(MediaItem.status == PreviewStatus.COMPLETED)
                completed_items = session.exec(statement).all()

                total_items = len(completed_items)
                logger.info(f"Found {total_items} completed items to verify")

                if total_items == 0:
                    return {"verified": 0, "moved_to_missing": 0}

                for item in completed_items:
                    verified_count += 1

                    # Check if BIF file exists
                    bif_exists = False
                    bif_path = None

                    if item.bundle_hash:
                        bif_path = self._get_bif_path(item.bundle_hash)
                        if bif_path and os.path.isfile(bif_path):
                            bif_exists = True
                        else:
                            # Log first few failures for debugging
                            if moved_count < 5:
                                logger.debug(f"BIF not found: {bif_path} (exists: {os.path.exists(bif_path) if bif_path else 'no path'})")
                    else:
                        logger.debug(f"Item {item.id} ({item.title}) has no bundle_hash")

                    # If BIF doesn't exist, track for moving
                    if not bif_exists:
                        missing_items.append((item, bif_path))
                        moved_count += 1

                # Calculate percentage that would be moved
                percentage_missing = (moved_count / total_items) * 100 if total_items > 0 else 0

                # Failsafe: if more than 50% would be moved, require confirmation
                if percentage_missing > 50 and not confirm_threshold:
                    logger.warning(f"Verification would move {moved_count}/{total_items} ({percentage_missing:.1f}%) items to MISSING. Requires confirmation.")
                    # Don't commit changes, return warning
                    return {
                        "verified": verified_count,
                        "moved_to_missing": 0,
                        "requires_confirmation": True,
                        "percentage": percentage_missing,
                        "would_move": moved_count,
                        "sample_paths": [path for _, path in missing_items[:3]]  # Show sample paths for debugging
                    }

                # Actually move items to MISSING
                for item, bif_path in missing_items:
                    logger.warning(f"BIF missing for completed item {item.id} ({item.title}): {bif_path}")
                    item.status = PreviewStatus.MISSING
                    item.progress = 0
                    session.add(item)

                session.commit()

            logger.info(f"Verification complete: {verified_count} items checked, {moved_count} moved to MISSING")
            return {"verified": verified_count, "moved_to_missing": moved_count}

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"verified": 0, "moved_to_missing": 0, "error": str(e)}

    def cleanup_stale_items(self):
        """Reset stale PROCESSING/QUEUED items to MISSING.

        This handles items that were left in PROCESSING or QUEUED status when:
        - The application was restarted
        - Workers crashed or were killed
        - Errors occurred that weren't properly handled

        Also checks for broken symlinks and marks them as MEDIA_MISSING.

        Returns:
            dict: Counts of items reset by status
        """
        logger.info("Cleaning up stale PROCESSING/QUEUED items...")

        try:
            import json
            with Session(engine) as session:
                # Get all PROCESSING and QUEUED items
                statement = select(MediaItem).where(
                    col(MediaItem.status).in_([PreviewStatus.PROCESSING, PreviewStatus.QUEUED])
                )
                stale_items = session.exec(statement).all()

                processing_count = 0
                queued_count = 0
                media_missing_count = 0

                for item in stale_items:
                    if item.status == PreviewStatus.PROCESSING:
                        processing_count += 1
                    else:
                        queued_count += 1

                    # Check if media files exist (detect broken symlinks)
                    media_exists = False
                    if item.media_parts_info:
                        try:
                            parts = json.loads(item.media_parts_info)
                            # Check if at least one part exists
                            for part in parts:
                                file_path = part.get('file_path')
                                if file_path and os.path.exists(file_path):
                                    media_exists = True
                                    break
                        except (json.JSONDecodeError, KeyError):
                            # If we can't parse media_parts_info, check file_path as fallback
                            if item.file_path and os.path.exists(item.file_path):
                                media_exists = True
                    elif item.file_path and os.path.exists(item.file_path):
                        # Fallback to file_path for backwards compatibility
                        media_exists = True

                    if not media_exists:
                        # Media file doesn't exist (broken symlink or deleted)
                        item.status = PreviewStatus.MEDIA_MISSING
                        item.progress = 0
                        item.error_message = "Media file not found (broken symlink or deleted)"
                        media_missing_count += 1
                        logger.debug(f"Media missing for item {item.id} ({item.title})")
                    else:
                        # Reset to MISSING for re-processing
                        item.status = PreviewStatus.MISSING
                        item.progress = 0
                        # Update timestamp so it gets re-queued soon
                        item.updated_at = datetime.utcnow()

                    session.add(item)

                session.commit()

                total_reset = processing_count + queued_count
                if total_reset > 0:
                    logger.info(f"Reset {total_reset} stale items (PROCESSING: {processing_count}, QUEUED: {queued_count}, MEDIA_MISSING: {media_missing_count})")
                else:
                    logger.debug("No stale items found")

                return {
                    "total_reset": total_reset,
                    "processing_reset": processing_count,
                    "queued_reset": queued_count,
                    "media_missing": media_missing_count
                }

        except Exception as e:
            logger.error(f"Stale item cleanup failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def detect_priority_items(self, plex, missing_rating_keys: Optional[set[int]] = None) -> dict[int, PriorityInfo]:
        """
        Detect scored priority items from Plex DB watch history and selected Plex hubs.
        """
        history_limit = 50
        try:
            with Session(engine) as session:
                settings = session.get(AppSettings, 1)
                if settings:
                    history_limit = settings.priority_history_limit
        except Exception as e:
            logger.debug(f"Could not load priority settings: {e}")

        try:
            if missing_rating_keys is None:
                with Session(engine) as session:
                    missing_rating_keys = set(
                        session.exec(
                            select(MediaItem.id).where(MediaItem.status == PreviewStatus.MISSING)
                        ).all()
                    )

            priority_infos = build_priority_infos(
                watch_events=self._read_priority_watch_events(history_limit),
                episodes=self._read_priority_episode_rows(),
                missing_rating_keys=missing_rating_keys,
                now=datetime.utcnow().replace(tzinfo=timezone.utc),
                hub_items_by_title=self._collect_priority_hub_items(plex),
                on_deck_rating_keys=self._collect_on_deck_rating_keys(plex),
            )

            logger.info(f"Detected {len(priority_infos)} scored priority items total")
            return priority_infos

        except Exception as e:
            logger.error(f"Priority detection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _plex_db_path(self) -> str:
        return os.path.join(
            self.config.plex_config_folder,
            "Plug-in Support",
            "Databases",
            "com.plexapp.plugins.library.db",
        )

    def _read_priority_watch_events(self, history_limit: int) -> list[WatchEvent]:
        db_path = self._plex_db_path()
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT account_id, rating_key, show_id, season_index, episode_index, viewed_at
                FROM (
                    SELECT
                        miv.account_id AS account_id,
                        ep.id AS rating_key,
                        show.id AS show_id,
                        season.'index' AS season_index,
                        ep.'index' AS episode_index,
                        miv.viewed_at AS viewed_at,
                        ROW_NUMBER() OVER (
                            PARTITION BY miv.account_id
                            ORDER BY miv.viewed_at DESC
                        ) AS row_num
                    FROM metadata_item_views miv
                    JOIN metadata_items ep ON miv.guid = ep.guid
                    JOIN metadata_items season ON ep.parent_id = season.id
                    JOIN metadata_items show ON season.parent_id = show.id
                    WHERE miv.account_id IS NOT NULL
                    AND ep.metadata_type = 4
                )
                WHERE row_num <= ?
                ORDER BY account_id, viewed_at DESC
                """,
                (history_limit,),
            )
            return [
                WatchEvent(
                    account_id=int(account_id),
                    rating_key=int(rating_key),
                    show_id=int(show_id),
                    season_index=int(season_index or 0),
                    episode_index=int(episode_index or 0),
                    viewed_at=datetime.fromtimestamp(int(viewed_at), tz=timezone.utc),
                )
                for account_id, rating_key, show_id, season_index, episode_index, viewed_at in cursor.fetchall()
            ]
        finally:
            conn.close()

    def _read_priority_episode_rows(self) -> list[EpisodeRow]:
        db_path = self._plex_db_path()
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ep.id, show.id, season.'index', ep.'index'
                FROM metadata_items ep
                JOIN metadata_items season ON ep.parent_id = season.id
                JOIN metadata_items show ON season.parent_id = show.id
                WHERE ep.metadata_type = 4
                """
            )
            return [
                EpisodeRow(
                    rating_key=int(rating_key),
                    show_id=int(show_id),
                    season_index=int(season_index or 0),
                    episode_index=int(episode_index or 0),
                )
                for rating_key, show_id, season_index, episode_index in cursor.fetchall()
            ]
        finally:
            conn.close()

    def _collect_on_deck_rating_keys(self, plex) -> set[int]:
        try:
            return {
                int(item.ratingKey)
                for item in retry_plex_call(plex.library.onDeck)
                if getattr(item, "ratingKey", None) is not None
            }
        except Exception as e:
            logger.warning(f"Failed to collect on-deck priority items: {e}")
            return set()

    def _collect_priority_hub_items(self, plex) -> dict[str, list[HubItem]]:
        hub_items_by_title: dict[str, list[HubItem]] = {}
        try:
            sections = retry_plex_call(plex.library.sections)
        except Exception as e:
            logger.warning(f"Failed to list Plex sections for hub priority: {e}")
            return hub_items_by_title

        for section in sections:
            try:
                hubs = retry_plex_call(section.hubs)
            except Exception as e:
                logger.warning(f"Failed to read hubs for {getattr(section, 'title', 'unknown')}: {e}")
                continue

            for hub in hubs:
                hub_title = getattr(hub, "title", "") or ""
                items_attr = getattr(hub, "items", None)
                try:
                    raw_items = items_attr() if callable(items_attr) else (items_attr or [])
                except Exception as e:
                    logger.warning(f"Failed to read hub items for {hub_title}: {e}")
                    continue

                converted_items = []
                for item in raw_items[:100]:
                    try:
                        rating_key = int(getattr(item, "ratingKey"))
                    except (TypeError, ValueError):
                        continue

                    item_type = getattr(item, "type", "")
                    show_id = None
                    season_index = None
                    if item_type == "season":
                        try:
                            show_id = int(getattr(item, "parentRatingKey"))
                        except (TypeError, ValueError):
                            show_id = None
                        season_index = getattr(item, "index", None)

                    converted_items.append(
                        HubItem(
                            rating_key=rating_key,
                            item_type=item_type,
                            title=getattr(item, "title", ""),
                            show_id=show_id,
                            season_index=season_index,
                        )
                    )

                if converted_items:
                    hub_items_by_title.setdefault(hub_title, []).extend(converted_items)

        return hub_items_by_title

    def _detect_priority_items_single_user(self, plex, history_limit: int) -> set:
        """
        Detect priority items for the current authenticated user.

        Args:
            plex: PlexServer instance
            history_limit: Number of recent history items to check

        Returns:
            set: Set of Plex rating keys (int) that are priority items
        """
        priority_keys = set()

        # Step 1: Get on-deck items
        logger.debug("Querying on-deck items...")
        try:
            on_deck_items = retry_plex_call(plex.library.onDeck)
            for item in on_deck_items:
                if item.ratingKey is not None:
                    priority_keys.add(int(item.ratingKey))
                    logger.debug(f"Priority (on-deck): {item.title} ({item.ratingKey})")
                else:
                    logger.debug(f"Skipping on-deck item with no ratingKey: {item.title}")
        except Exception as e:
            logger.warning(f"Failed to get on-deck items: {e}")

        # Step 2: Get recently played items
        logger.debug(f"Querying recently played items (last {history_limit})...")
        try:
            history_items = retry_plex_call(plex.library.history, maxresults=history_limit)

            # Track seasons we've seen to avoid duplicate season queries
            processed_seasons = set()

            for item in history_items:
                if item.ratingKey is None:
                    logger.debug(f"Skipping history item with no ratingKey: {getattr(item, 'title', 'Unknown')}")
                    continue

                priority_keys.add(int(item.ratingKey))
                logger.debug(f"Priority (history): {item.title} ({item.ratingKey})")

                # Step 3: If it's an episode, get all episodes from same season
                if item.type == 'episode':
                    try:
                        # Get parent season
                        season = retry_plex_call(item.season)
                        if season.ratingKey is None:
                            logger.debug(f"Skipping season with no ratingKey for episode: {item.title}")
                            continue

                        season_key = int(season.ratingKey)

                        # Only process each season once (optimization)
                        if season_key not in processed_seasons:
                            processed_seasons.add(season_key)

                            # Get all episodes in this season
                            episodes = retry_plex_call(season.episodes)
                            for ep in episodes:
                                if ep.ratingKey is not None:
                                    priority_keys.add(int(ep.ratingKey))

                            logger.debug(f"Priority (adjacent): Added {len(episodes)} episodes from {season.title}")
                    except Exception as e:
                        logger.warning(f"Failed to get adjacent episodes for {item.title}: {e}")
        except Exception as e:
            logger.warning(f"Failed to get history items: {e}")

        return priority_keys

    def _detect_priority_items_all_users(self, plex, history_limit: int) -> set:
        """
        Detect priority items for all users on the Plex server.

        Requires server owner/admin token to access all users' data.

        Args:
            plex: PlexServer instance
            history_limit: Number of recent history items to check per user

        Returns:
            set: Set of Plex rating keys (int) that are priority items
        """
        priority_keys = set()

        # Get all users on the server
        all_users = []
        try:
            # Try to get myPlex account (requires owner/admin)
            my_account = plex.myPlexAccount()

            # Get managed users (home users)
            try:
                users = my_account.users()
                logger.info(f"Found {len(users)} managed users on server")

                # Include the server owner
                all_users.append({
                    'username': my_account.username or 'Server Owner',
                    'id': my_account.id,
                    'is_owner': True
                })

                # Add managed users
                for u in users:
                    all_users.append({
                        'username': u.username or u.title,
                        'id': u.id,
                        'is_owner': False
                    })

            except Exception as e:
                logger.warning(f"Could not get managed users list: {e}")
                # Fallback to just the owner
                all_users.append({
                    'username': my_account.username or 'Server Owner',
                    'id': my_account.id,
                    'is_owner': True
                })

        except Exception as e:
            logger.warning(f"Could not access myPlex account (may not be server owner): {e}")
            logger.info("Falling back to single-user priority detection")
            return self._detect_priority_items_single_user(plex, history_limit)

        logger.info(f"Checking priority items for {len(all_users)} users")
        logger.debug(f"User list: {[u['username'] for u in all_users]}")

        # Track seasons globally to avoid duplicate processing across users
        global_processed_seasons = set()

        # Fetch watch history from database (more reliable than API for multi-user)
        logger.info("Fetching watch history from database for all users...")
        history_by_user = get_watch_history_from_database(self.config.plex_config_folder, history_limit)
        logger.info(f"Retrieved history for {len(history_by_user)} users from database")

        # Verify we got history for all expected users
        db_account_ids = set(history_by_user.keys())
        plex_account_ids = set(u['id'] for u in all_users)
        missing_users = plex_account_ids - db_account_ids
        if missing_users:
            logger.warning(f"No watch history found in database for {len(missing_users)} user(s): {missing_users}")
            logger.warning("These users may not have watched anything yet, or their account_id format differs")
        else:
            logger.info(f"✓ Found watch history for all {len(all_users)} users")

        # Process each user
        for user_info in all_users:
            username = user_info['username']
            logger.debug(f"Checking priority items for user: {username}")

            # Get on-deck items for this user
            try:
                # Note: plex.library.onDeck() returns results for all users when called with admin token
                # We can't filter by specific user easily with PlexAPI, so we get all on-deck items
                # This is actually beneficial as it includes all users' on-deck items
                if user_info['is_owner']:
                    on_deck_items = retry_plex_call(plex.library.onDeck)
                    for item in on_deck_items:
                        if item.ratingKey is not None:
                            priority_keys.add(int(item.ratingKey))
                            logger.debug(f"Priority (on-deck): {item.title} ({item.ratingKey})")
                # For non-owner users, on-deck is already included above
            except Exception as e:
                logger.warning(f"Failed to get on-deck items for {username}: {e}")

            # Get recently watched items for this user
            try:
                # Get this user's history from database cache (returns rating keys)
                user_history_rating_keys = history_by_user.get(user_info['id'], [])
                logger.debug(f"Found {len(user_history_rating_keys)} history items for user {username}")

                for metadata_item_id, viewed_at in user_history_rating_keys:
                    # Add the rating key to priority set
                    priority_keys.add(int(metadata_item_id))
                    logger.debug(f"Priority (history, {username}): rating_key={metadata_item_id}")

                    # For episodes, fetch the item to get season and all adjacent episodes
                    try:
                        # Fetch the item from Plex to determine type and get season info
                        item = retry_plex_call(plex.fetchItem, int(metadata_item_id))

                        if item.type == 'episode':
                            try:
                                season = retry_plex_call(item.season)
                                if season.ratingKey is None:
                                    logger.debug(f"Skipping season with no ratingKey for episode: {item.title}")
                                    continue

                                season_key = int(season.ratingKey)

                                # Only process each season once globally (across all users)
                                if season_key not in global_processed_seasons:
                                    global_processed_seasons.add(season_key)

                                    # Get all episodes in this season
                                    episodes = retry_plex_call(season.episodes)
                                    for ep in episodes:
                                        if ep.ratingKey is not None:
                                            priority_keys.add(int(ep.ratingKey))

                                    logger.debug(f"Priority (adjacent, {username}): Added {len(episodes)} episodes from {season.title}")
                            except Exception as e:
                                logger.warning(f"Failed to get adjacent episodes for {username}/rating_key={metadata_item_id}: {e}")
                    except Exception as e:
                        # Item might have been deleted, just skip
                        logger.debug(f"Could not fetch item {metadata_item_id} for user {username}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Failed to process history for user {username}: {e}")
                # Continue with other users even if one fails

        return priority_keys

    def _process_sync_item(self, section_title, item_key, title, item_type, bundle_hash, added_at,
                          media_parts, bundle_hash_map, priority_item_keys, db_item):
        """
        Process a single item during sync. Returns tuple of (item_to_add_or_update, stat_type).
        stat_type is one of: 'movies_added', 'movies_updated', 'episodes_added', 'episodes_updated', None
        """
        import json

        # Prepare media parts JSON
        media_parts_json = None
        if media_parts:
            parts_data = [
                {
                    "file_path": file_path,
                    "bundle_hash": bundle_hash_part,
                    "bif_path": self._get_bif_path(bundle_hash_part) if bundle_hash_part else None
                }
                for file_path, bundle_hash_part in media_parts
            ]
            media_parts_json = json.dumps(parts_data)

        # Check if BIF exists using pre-scanned map
        bif_exists = bundle_hash and bundle_hash in bundle_hash_map

        # Check if media files exist (detect broken symlinks)
        media_exists = False
        if media_parts:
            for file_path, _ in media_parts:
                if file_path and os.path.exists(file_path):
                    media_exists = True
                    break

        if not db_item:
            # Create new item
            if not media_exists:
                initial_status = PreviewStatus.MEDIA_MISSING
                initial_progress = 0
            elif bif_exists:
                initial_status = PreviewStatus.COMPLETED
                initial_progress = 100
            else:
                initial_status = PreviewStatus.MISSING
                initial_progress = 0

            new_item = MediaItem(
                id=int(item_key),
                title=title,
                media_type=MediaType.MOVIE if item_type == 'movie' else MediaType.EPISODE,
                library_name=section_title,
                status=initial_status,
                progress=initial_progress,
                bundle_hash=bundle_hash,
                media_parts_info=media_parts_json,
                added_at=added_at if added_at else datetime.utcnow(),
                error_message="Media file not found (broken symlink or deleted)" if not media_exists else None,
            )
            self._apply_priority_info_to_item(
                new_item,
                self._priority_info_for_key(priority_item_keys, int(item_key)),
            )

            stat_type = 'movies_added' if item_type == 'movie' else 'episodes_added'
            return (new_item, stat_type)
        else:
            # Update existing item
            item_updated = False

            # Update priority status
            priority_info = self._priority_info_for_key(priority_item_keys, int(item_key))
            if self._apply_priority_info_to_item(db_item, priority_info):
                item_updated = True
                if priority_info:
                    logger.debug(f"Item {item_key} ({title}) marked as priority score={priority_info.score}")

            # Update hash if missing
            if not db_item.bundle_hash and bundle_hash:
                db_item.bundle_hash = bundle_hash

            # Update media_parts_info
            if media_parts_json:
                db_item.media_parts_info = media_parts_json

            # Update added_at if differs
            if added_at and db_item.added_at != added_at:
                db_item.added_at = added_at

            # Check if media files exist - if not, mark as MEDIA_MISSING
            if not media_exists:
                if db_item.status != PreviewStatus.MEDIA_MISSING:
                    logger.info(f"Item {item_key} ({title}) has missing media files - marking as MEDIA_MISSING")
                    db_item.status = PreviewStatus.MEDIA_MISSING
                    db_item.progress = 0
                    db_item.error_message = "Media file not found (broken symlink or deleted)"
                    item_updated = True
            else:
                # Media exists - check BIF status
                any_missing_bif = False
                if media_parts_json:
                    parts_data = json.loads(media_parts_json)
                    any_missing_bif = any(part.get('bif_path') is None for part in parts_data)

                if db_item.status == PreviewStatus.COMPLETED and any_missing_bif:
                    logger.info(f"Item {item_key} ({title}) is completed but has new parts without BIF files - resetting to missing")
                    db_item.status = PreviewStatus.MISSING
                    db_item.progress = 0
                    db_item.updated_at = datetime.utcnow()
                    item_updated = True
                elif db_item.status != PreviewStatus.COMPLETED and bif_exists and not any_missing_bif:
                    db_item.status = PreviewStatus.COMPLETED
                    db_item.progress = 100
                    item_updated = True
                elif db_item.status == PreviewStatus.MEDIA_MISSING:
                    logger.info(f"Item {item_key} ({title}) media file now exists - resetting to MISSING")
                    db_item.status = PreviewStatus.MISSING
                    db_item.progress = 0
                    db_item.error_message = None
                    db_item.updated_at = datetime.utcnow()
                    item_updated = True

            stat_type = None
            if item_updated:
                stat_type = 'movies_updated' if db_item.media_type == MediaType.MOVIE else 'episodes_updated'

            return (db_item, stat_type)

    def sync_library(self):
        logger.info("Syncing library with Plex...")
        self.sync_in_progress = True
        sync_start_time = time.time()
        try:
            # Step 0: Validate mount paths before syncing
            mount_valid, missing_paths = self.validate_mount_paths()
            if not mount_valid:
                logger.error(f"⚠️⚠️⚠️  MOUNT VALIDATION FAILED - Sync aborted! ⚠️⚠️⚠️")
                logger.error(f"Missing/inaccessible paths: {', '.join(missing_paths)}")
                logger.error("Please ensure your media mount is ready before syncing")
                logger.error("Sync will be retried on next interval")
                # Pause queue to prevent processing with unmounted media
                if not self.paused:
                    logger.error("Auto-pausing queue due to mount failure")
                    self.pause()
                self.sync_in_progress = False
                return

            # Track sync statistics
            sync_stats = {
                "movies_added": 0,
                "movies_updated": 0,
                "movies_deleted": 0,
                "episodes_added": 0,
                "episodes_updated": 0,
                "episodes_deleted": 0,
            }

            # Step 1: Scan filesystem once for all BIF files
            logger.info("Scanning filesystem for BIF files...")
            bif_scan_start = time.time()
            bundle_hash_map = self._scan_filesystem_for_bifs()
            logger.info(f"Found {len(bundle_hash_map)} existing BIF files in {time.time() - bif_scan_start:.2f}s")

            # Step 2: Get items from Plex
            logger.info("Fetching items from Plex...")
            plex_fetch_start = time.time()
            plex = plex_server(self.config)

            # Check if database-only sync is enabled
            use_database_sync = True  # Default to True for speed
            try:
                with Session(engine) as settings_session:
                    settings = settings_session.get(AppSettings, 1)
                    if settings and hasattr(settings, 'use_database_sync'):
                        use_database_sync = settings.use_database_sync
            except Exception as e:
                logger.debug(f"Could not load use_database_sync setting, defaulting to True: {e}")

            sections = get_library_sections(plex, self.config, database_only=use_database_sync)

            # Collect all items from Plex into a list
            all_items = []
            plex_item_ids = set()
            for section, items in sections:
                for item_key, title, item_type, bundle_hash, added_at in items:
                    plex_item_ids.add(int(item_key))
                    all_items.append((section.title, item_key, title, item_type, bundle_hash, added_at))

            logger.info(f"Fetched {len(all_items)} items from Plex in {time.time() - plex_fetch_start:.2f}s")

            # Step 2.5: Detect scored priority items from Plex
            missing_rating_keys = {
                int(item_key)
                for _, item_key, _, _, bundle_hash, _ in all_items
                if not bundle_hash or bundle_hash not in bundle_hash_map
            }
            priority_item_keys = self.detect_priority_items(plex, missing_rating_keys=missing_rating_keys)
            logger.debug(f"Priority detection complete: {len(priority_item_keys)} items")

            # Step 3: Batch query all media parts at once
            logger.info("Batch querying media parts from Plex database...")
            batch_query_start = time.time()
            all_rating_keys = [int(item[1]) for item in all_items]
            media_parts_map = get_all_media_parts_batch(self.config.plex_config_folder, all_rating_keys)
            logger.info(f"Queried media parts for {len(all_rating_keys)} items in {time.time() - batch_query_start:.2f}s")

            # Step 4: Process items in parallel with threading
            logger.info(f"Processing {len(all_items)} items with parallel threads...")
            process_start = time.time()

            with Session(engine) as session:
                # Load all existing items from database into a dict for fast lookup
                from sqlmodel import select
                db_items_list = session.exec(select(MediaItem)).all()
                db_items_dict = {item.id: item for item in db_items_list}
                db_item_ids = set(db_items_dict.keys())

                # Process items in parallel
                items_to_update = []
                stats_lock = threading.Lock()

                def process_item_wrapper(item_data):
                    section_title, item_key, title, item_type, bundle_hash, added_at = item_data
                    media_parts = media_parts_map.get(int(item_key), [])
                    db_item = db_items_dict.get(int(item_key))

                    return self._process_sync_item(
                        section_title, item_key, title, item_type, bundle_hash, added_at,
                        media_parts, bundle_hash_map, priority_item_keys, db_item
                    )

                # Use ThreadPoolExecutor for parallel processing
                max_workers = min(32, len(all_items))  # Limit to 32 threads
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_item_wrapper, item): item for item in all_items}

                    for future in as_completed(futures):
                        try:
                            item, stat_type = future.result()
                            if item:
                                items_to_update.append(item)
                                if stat_type:
                                    with stats_lock:
                                        sync_stats[stat_type] += 1
                        except Exception as e:
                            logger.error(f"Error processing item: {e}")

                # Batch add all items to session
                for item in items_to_update:
                    session.add(item)

                logger.info(f"Processed {len(all_items)} items in {time.time() - process_start:.2f}s")

                # Step 5: Clean up orphaned items (items in DB but no longer in Plex)
                logger.debug(f"Found {len(plex_item_ids)} items in Plex")
                logger.debug(f"Found {len(db_item_ids)} items in database")

                # Find orphaned items (in DB but not in Plex)
                orphaned_ids = db_item_ids - plex_item_ids

                if orphaned_ids:
                    logger.info(f"Found {len(orphaned_ids)} orphaned items no longer in Plex - removing from database")

                    # Delete orphaned items
                    for orphaned_id in orphaned_ids:
                        orphaned_item = session.get(MediaItem, orphaned_id)
                        if orphaned_item:
                            logger.debug(f"Removing orphaned item: {orphaned_id} ({orphaned_item.title})")
                            # Track deletion in sync stats
                            if orphaned_item.media_type == MediaType.MOVIE:
                                sync_stats["movies_deleted"] += 1
                            else:
                                sync_stats["episodes_deleted"] += 1
                            session.delete(orphaned_item)

                    logger.info(f"Removed {len(orphaned_ids)} orphaned items from database")
                else:
                    logger.debug("No orphaned items found")

                # Sanity check: Detect mass MEDIA_MISSING marking (likely mount issue)
                # Count how many items would be newly marked as MEDIA_MISSING
                newly_media_missing = 0
                total_checked = 0
                for item in chain(session.new, session.dirty):
                    if isinstance(item, MediaItem):
                        # Check if this is a newly marked MEDIA_MISSING item
                        if hasattr(item, '__dict__') and '_sa_instance_state' in item.__dict__:
                            # Get original state from session
                            history = session.identity_map.get((MediaItem, item.id))
                            if history and getattr(history, 'status', None) != PreviewStatus.MEDIA_MISSING and item.status == PreviewStatus.MEDIA_MISSING:
                                newly_media_missing += 1
                        total_checked += 1

                # Get threshold from settings
                mount_failure_threshold = 50.0  # Default
                try:
                    settings = session.get(AppSettings, 1)
                    if settings and hasattr(settings, 'mount_failure_threshold'):
                        mount_failure_threshold = settings.mount_failure_threshold
                except:
                    pass

                # Calculate percentage
                if total_checked > 0:
                    missing_percentage = (newly_media_missing / total_checked) * 100
                    if missing_percentage > mount_failure_threshold:
                        logger.error(f"⚠️⚠️⚠️  SANITY CHECK FAILED - Sync aborted! ⚠️⚠️⚠️")
                        logger.error(f"{newly_media_missing}/{total_checked} items ({missing_percentage:.1f}%) would be marked as MEDIA_MISSING")
                        logger.error(f"This exceeds the threshold of {mount_failure_threshold}% and likely indicates a mount failure")
                        logger.error("Changes rolled back. Please check your media mounts and retry sync")
                        # Rollback instead of commit
                        session.rollback()
                        # Pause queue
                        if not self.paused:
                            logger.error("Auto-pausing queue due to suspected mount failure")
                            self.pause()
                        return

                session.commit()
            
            # Save last sync time and summary to AppSettings
            import json
            new_last_sync_time = None
            with Session(engine) as settings_session:
                settings = settings_session.get(AppSettings, 1)
                if not settings:
                    settings = AppSettings(id=1) # Should exist if config was loaded
                settings.last_sync_time = datetime.utcnow()
                settings.last_sync_summary = json.dumps(sync_stats)
                settings_session.add(settings)
                settings_session.commit()
                new_last_sync_time = settings.last_sync_time # Extract value while session is open
            self.last_sync_time = new_last_sync_time # Assign after session is closed

            total_sync_time = time.time() - sync_start_time
            logger.info(f"Library sync complete in {total_sync_time:.2f}s. {sync_stats}")
        except Exception as e:
            logger.error(f"Sync failed: {e}")
        finally:
            self.sync_in_progress = False

    def process_queue(self):
        if self.paused:
            return

        # Check rate limit status
        is_rate_limited = self._check_rate_limit()

        # Validate mount paths before processing
        mount_valid, missing_paths = self.validate_mount_paths()
        if not mount_valid:
            logger.warning(f"Mount validation failed during queue processing: {', '.join(missing_paths)}")
            logger.warning("Skipping queue processing until mounts are available")
            # Pause queue to prevent repeated errors
            if not self.paused:
                logger.error("Auto-pausing queue due to mount validation failure")
                self.pause()
            return

        # Fetch batch of items - match the number of available workers
        # This ensures we queue exactly as many items as we can process in parallel
        if self.config and self.worker_pool:
            batch_size = self.config.gpu_threads + self.config.cpu_threads
        else:
            batch_size = 10  # Fallback if config not loaded yet

        # If rate-limited, fetch more items to find exempt ones
        fetch_limit = batch_size * 5 if is_rate_limited else batch_size

        with Session(engine) as session:
            # Prioritize: Status is MISSING only (not QUEUED - those are already in the worker queue)
            # Sort by:
            # 1. priority_score desc (ranked automatic priority)
            # 2. updated_at desc (Most recently changed first)
            # 3. queue_order asc (Manual prioritization)
            statement = select(MediaItem).where(
                MediaItem.status == PreviewStatus.MISSING
            ).order_by(*self._priority_order_by()).limit(fetch_limit)

            items = session.exec(statement).all()

            if not items:
                return

            # If rate-limited, filter to only include exempt items
            if is_rate_limited:
                original_count = len(items)
                exempt_items = []
                rate_limited_items = []

                for item in items:
                    if self._is_item_rate_limit_exempt(item):
                        exempt_items.append(item)
                    else:
                        rate_limited_items.append(item)

                if exempt_items:
                    logger.info(f"Rate limited: Processing {len(exempt_items)} exempt items, skipping {len(rate_limited_items)} rate-limited items")
                    items = exempt_items[:batch_size]  # Only process up to batch_size
                else:
                    logger.debug(f"Rate limited: No exempt items found in {original_count} candidates. Waiting for quota reset.")
                    return

            # Log the order of items being processed for debugging
            if items:
                logger.debug(f"Processing batch of {len(items)} items (priority first, then most recently updated):")
                for idx, item in enumerate(items[:3]):  # Show first 3 items
                    updated_date = item.updated_at.strftime('%Y-%m-%d %H:%M') if item.updated_at else 'unknown'
                    priority_flag = " [PRIORITY]" if item.is_priority else ""
                    exempt_flag = " [EXEMPT]" if self._is_item_rate_limit_exempt(item) else ""
                    logger.debug(f"  {idx+1}. {item.title} (updated: {updated_date}){priority_flag}{exempt_flag}")

            # Prepare for worker pool
            # worker_pool.process_items expects List[tuple(key, title, type)]
            process_list = []
            for item in items:
                # Update status to QUEUED if it was MISSING
                if item.status == PreviewStatus.MISSING:
                    item.status = PreviewStatus.QUEUED
                    session.add(item)

                process_list.append((str(item.id), item.title, item.media_type))

            session.commit()
        
        if not process_list:
            return

        logger.debug(f"Processing batch of {len(process_list)} items")

        # Instantiate DbProgressManager
        pm = DbProgressManager()

        # Get Plex server again (fresh connection)
        plex = plex_server(self.config)

        # Define callback to fetch more items when queue runs low
        def fetch_more_items():
            """Fetch more items from database when queue is running low."""
            if self.paused:
                return []

            # Check rate limit status for filtering
            current_rate_limited = self._check_rate_limit()
            current_fetch_limit = batch_size * 5 if current_rate_limited else batch_size

            with Session(engine) as session:
                # Fetch items that are MISSING only (QUEUED items are already in the worker queue)
                # Sort by priority first, then updated_at, then manual queue_order
                statement = select(MediaItem).where(
                    MediaItem.status == PreviewStatus.MISSING
                ).order_by(*self._priority_order_by()).limit(current_fetch_limit)

                items = session.exec(statement).all()

                if not items:
                    return []

                # If rate-limited, filter to only include exempt items
                if current_rate_limited:
                    exempt_items = [item for item in items if self._is_item_rate_limit_exempt(item)]
                    if exempt_items:
                        logger.debug(f"Rate limited: Fetching {len(exempt_items)} exempt items out of {len(items)} candidates")
                        items = exempt_items[:batch_size]
                    else:
                        logger.debug(f"Rate limited: No exempt items found in {len(items)} candidates")
                        return []

                # Mark items as QUEUED and prepare for processing
                new_items = []
                for item in items:
                    if item.status == PreviewStatus.MISSING:
                        item.status = PreviewStatus.QUEUED
                        session.add(item)
                    new_items.append((str(item.id), item.title, item.media_type))

                session.commit()

                if new_items:
                    logger.info(f"Dynamically fetched {len(new_items)} more items for processing")

                return new_items

        # Pause handler for Plex connection errors
        def pause_handler():
            """Pause queue when Plex connection error is detected."""
            if not self.paused:
                self.paused = True
                logger.error("⚠️⚠️⚠️  QUEUE PAUSED DUE TO PLEX CONNECTION ERROR ⚠️⚠️⚠️")
                logger.error("Please check your Plex server status and network connectivity")
                logger.error("Resume processing from the web interface once Plex is accessible")

        # Stop condition: Only stop when paused (rate limit is handled by filtering exempt items)
        def should_stop():
            if self.paused:
                return True
            # When rate-limited, we continue but only process exempt items
            # The actual filtering happens in fetch_more_items
            return False

        # Run processing with dynamic queue refilling
        self.worker_pool.process_items(
            process_list,
            self.config,
            plex,
            pm,
            stop_condition=should_stop,
            fetch_more_items=fetch_more_items,
            pause_handler=pause_handler
        )
        
        # Post-processing: Ensure all items that should be completed are marked as completed
        # This is a safety net in case progress updates were missed
        with Session(engine) as session:
            for item_id, _, _ in process_list:
                item = session.get(MediaItem, int(item_id))
                if item:
                    # Check if BIF file exists for this item
                    bif_exists = False
                    if item.bundle_hash:
                        bif_path = self._get_bif_path(item.bundle_hash)
                        if bif_path and os.path.isfile(bif_path):
                            bif_exists = True

                    # If BIF exists but status is not COMPLETED/FAILED, mark it as completed
                    if bif_exists and item.status not in [PreviewStatus.COMPLETED, PreviewStatus.FAILED, PreviewStatus.SLOW_FAILED]:
                        logger.debug(f"Post-processing: Marking item {item_id} ({item.title}) as COMPLETED (BIF file exists)")
                        item.status = PreviewStatus.COMPLETED
                        item.progress = 100
                        session.add(item)
                    # If progress is 100% but status is not COMPLETED/FAILED/SLOW_FAILED, mark it as completed
                    # (Don't override FAILED or SLOW_FAILED status even if progress is 100%)
                    elif item.progress >= 100 and item.status not in [PreviewStatus.COMPLETED, PreviewStatus.FAILED, PreviewStatus.SLOW_FAILED]:
                        logger.debug(f"Post-processing: Marking item {item_id} ({item.title}) as COMPLETED (progress=100%)")
                        item.status = PreviewStatus.COMPLETED
                        item.progress = 100
                        session.add(item)
                    # If item was processing but progress is still 0, it might have failed
                    elif item.status == PreviewStatus.PROCESSING and item.progress == 0:
                        # Reset to QUEUED so it can be retried
                        logger.debug(f"Post-processing: Item {item_id} ({item.title}) was processing but no progress made, resetting to QUEUED")
                        item.status = PreviewStatus.QUEUED
                        session.add(item)

                    # Clear priority metadata for completed items to prevent stale priority data
                    if self._clear_completed_priority_metadata(item):
                        logger.debug(f"Post-processing: Clearing priority metadata for completed item {item_id} ({item.title})")
                        session.add(item)
            session.commit()

scheduler = Scheduler()
