import threading
import time
import logging
from typing import List, Tuple, Optional
from sqlmodel import Session, select, col
from loguru import logger

from ..config import Config, load_config
from ..plex_client import plex_server, get_library_sections
from ..worker import WorkerPool
from ..gpu_detection import detect_all_gpus
from ..utils import setup_working_directory, sanitize_path
from .database import engine, get_session
from .models import MediaItem, PreviewStatus, MediaType, AppSettings
import os
from datetime import datetime

class DbProgressManager:
    def __init__(self):
        # Track last update time per worker to throttle DB writes
        self.last_update = {}

    def init_workers(self, workers):
        pass

    def update_worker(self, worker_id, data: dict):
        item_key = data.get('item_key')
        if not item_key:
            return

        progress = data.get('progress_percent', 0)
        is_busy = data.get('is_busy', False)
        error_message = data.get('error_message')
        media_file = data.get('media_file')

        # Throttle DB updates to every 2 seconds per worker, unless finished (100%) or starting (0%)
        current_time = time.time()
        last_time = self.last_update.get(worker_id, 0)

        should_update = (
            progress == 0 or            # Always update on 0%
            progress >= 100 or          # Always update on 100%
            data.get('failed') or       # Always update on failed
            (current_time - last_time > 2.0 and is_busy) # Throttle if busy and not a final state
        )

        if should_update:
            self.last_update[worker_id] = current_time
            try:
                with Session(engine) as session:
                    item = session.get(MediaItem, int(item_key))
                    if item:
                        if data.get('failed'):
                            # Task failed - mark as FAILED regardless of progress
                            item.status = PreviewStatus.FAILED
                            item.progress = int(progress) if progress > 0 else item.progress
                            item.error_message = error_message
                            logger.info(f"Marked item {item_key} ({item.title}) as FAILED in database: {error_message}")
                        elif progress >= 100:
                            # Only mark as COMPLETED if not failed
                            item.status = PreviewStatus.COMPLETED
                            item.progress = 100
                            item.error_message = None  # Clear any previous error
                            # Set BIF path if we have a bundle hash
                            if item.bundle_hash:
                                from ..utils import sanitize_path
                                bundle_file = sanitize_path(f'{item.bundle_hash[0]}/{item.bundle_hash[1::1]}.bundle')
                                bundle_path = sanitize_path(os.path.join(
                                    scheduler.config.plex_config_folder if scheduler.config else '/config/plex',
                                    'Media', 'localhost', bundle_file
                                ))
                                bif_path = sanitize_path(os.path.join(bundle_path, 'Contents', 'Indexes', 'index-sd.bif'))
                                item.bif_path = bif_path
                            logger.info(f"Marked item {item_key} ({item.title}) as COMPLETED in database")
                        elif progress > 0:
                            # Only update to PROCESSING if not already COMPLETED
                            # (prevents race condition where completion update comes after this)
                            if item.status != PreviewStatus.COMPLETED:
                                item.status = PreviewStatus.PROCESSING
                            item.progress = int(progress)
                            # Store media file path if available
                            if media_file:
                                item.file_path = media_file
                        session.add(item)
                        session.commit()
            except Exception as e:
                logger.error(f"Failed to update DB progress for {item_key}: {e}")

    def update_main_progress(self, completed, total):
        pass

    def cleanup_workers(self):
        pass

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

    def trigger_sync(self):
        self.force_sync = True
        logger.info("Manual sync requested")
        self.sync_wake_event.set()  # Wake sync thread for immediate sync

    def pause(self):
        self.paused = True
        logger.info("Queue paused")

    def resume(self):
        self.paused = False
        logger.info("Queue resumed")

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

                # Check if sync is needed (every 12 hours or when forced)
                sync_needed = self.force_sync or \
                              (time.time() - self.last_sync_time.timestamp() > 12 * 3600 if self.last_sync_time else True)

                if sync_needed:
                    logger.info("Background sync: Starting library sync...")
                    self.sync_library()
                    self.last_sync_time = datetime.utcnow()
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
                
                logger.debug(f"Scheduler loop done. Sleeping for {self.config.scheduler_loop_interval}s.")
                
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
                analysis_triggered = False

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
                                cursor.execute("SELECT hash FROM metadata_items WHERE id = ?", (item.id,))
                                result = cursor.fetchone()
                                conn.close()

                                if result and result[0]:
                                    item.bundle_hash = result[0]
                                    logger.debug(f"Retrieved hash from Plex database for item {item.id}: {item.bundle_hash}")
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
                            analysis_triggered = True
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
                
                # Schedule sync if we triggered analysis
                if analysis_triggered:
                    logger.info("Scheduled forced sync to pick up new bundle hashes after analysis")
                    self.force_sync = True

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

            # Get all bundle hashes from Plex database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT hash FROM metadata_items WHERE hash IS NOT NULL")
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

    def sync_library(self):
        logger.debug("Syncing library with Plex...")
        try:
            # Step 1: Scan filesystem once for all BIF files
            bundle_hash_map = self._scan_filesystem_for_bifs()
            logger.debug(f"Found {len(bundle_hash_map)} existing BIF files in filesystem")

            # Step 2: Get items from Plex
            plex = plex_server(self.config)
            sections = get_library_sections(plex, self.config)

            with Session(engine) as session:
                for section, items in sections:
                    for item_key, title, item_type, bundle_hash, added_at in items:
                        # Check if exists
                        db_item = session.get(MediaItem, int(item_key))

                        # Check if BIF exists using pre-scanned map (no filesystem access)
                        bif_exists = bundle_hash and bundle_hash in bundle_hash_map

                        if not db_item:
                            # Create new
                            new_item = MediaItem(
                                id=int(item_key),
                                title=title,
                                media_type=MediaType.MOVIE if item_type == 'movie' else MediaType.EPISODE,
                                library_name=section.title,
                                status=PreviewStatus.COMPLETED if bif_exists else PreviewStatus.MISSING,
                                progress=100 if bif_exists else 0,
                                bundle_hash=bundle_hash,
                                added_at=added_at if added_at else datetime.utcnow()
                            )
                            session.add(new_item)
                        else:
                            # Update hash if missing
                            if not db_item.bundle_hash and bundle_hash:
                                db_item.bundle_hash = bundle_hash
                                session.add(db_item)

                            # Update added_at if differs (sync)
                            if added_at and db_item.added_at != added_at:
                                db_item.added_at = added_at
                                session.add(db_item)

                            # If existing item is missing/queued/processing but BIF exists, mark completed
                            if db_item.status != PreviewStatus.COMPLETED and bif_exists:
                                db_item.status = PreviewStatus.COMPLETED
                                db_item.progress = 100
                                session.add(db_item)

                session.commit()
            
            # Save last sync time to AppSettings
            new_last_sync_time = None
            with Session(engine) as settings_session:
                settings = settings_session.get(AppSettings, 1)
                if not settings:
                    settings = AppSettings(id=1) # Should exist if config was loaded
                settings.last_sync_time = datetime.utcnow()
                settings_session.add(settings)
                settings_session.commit()
                new_last_sync_time = settings.last_sync_time # Extract value while session is open
            self.last_sync_time = new_last_sync_time # Assign after session is closed
            
            logger.info("Library sync complete.")
        except Exception as e:
            logger.error(f"Sync failed: {e}")

    def process_queue(self):
        if self.paused:
            return

        # Fetch batch of items - match the number of available workers
        # This ensures we queue exactly as many items as we can process in parallel
        if self.config and self.worker_pool:
            batch_size = self.config.gpu_threads + self.config.cpu_threads
        else:
            batch_size = 10  # Fallback if config not loaded yet
        
        with Session(engine) as session:
            # Prioritize: Status is MISSING or QUEUED.
            # Sort by added_at desc (Newest first), then by queue_order for manual prioritization
            # This ensures newest items are ALWAYS processed first, with queue_order as tiebreaker
            statement = select(MediaItem).where(
                col(MediaItem.status).in_([PreviewStatus.MISSING, PreviewStatus.QUEUED])
            ).order_by(MediaItem.added_at.desc(), MediaItem.queue_order.asc()).limit(batch_size)

            items = session.exec(statement).all()

            if not items:
                return

            # Log the order of items being processed for debugging
            if items:
                logger.debug(f"Processing batch of {len(items)} items (newest first):")
                for idx, item in enumerate(items[:3]):  # Show first 3 items
                    added_date = item.added_at.strftime('%Y-%m-%d') if item.added_at else 'unknown'
                    logger.debug(f"  {idx+1}. {item.title} (added: {added_date})")

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
        
        # Run processing
        # Note: This blocks until batch is done
        self.worker_pool.process_items(
            process_list, 
            self.config, 
            plex, 
            pm,
            stop_condition=lambda: self.paused
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
                    if bif_exists and item.status not in [PreviewStatus.COMPLETED, PreviewStatus.FAILED]:
                        logger.debug(f"Post-processing: Marking item {item_id} ({item.title}) as COMPLETED (BIF file exists)")
                        item.status = PreviewStatus.COMPLETED
                        item.progress = 100
                        session.add(item)
                    # If progress is 100% but status is not COMPLETED/FAILED, mark it as completed
                    # (Don't override FAILED status even if progress is 100%)
                    elif item.progress >= 100 and item.status not in [PreviewStatus.COMPLETED, PreviewStatus.FAILED]:
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
            session.commit()

scheduler = Scheduler()
