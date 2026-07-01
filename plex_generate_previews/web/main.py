import os
import requests
import uuid
import hashlib
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, Form, HTTPException, status, Cookie, Body
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select, func, col, delete, or_, update
from sqlalchemy import case
from sqlalchemy.types import Float
from typing import Optional, List
from pydantic import BaseModel
from itsdangerous import URLSafeTimedSerializer
from loguru import logger

from .database import create_db_and_tables, get_session, engine
from .models import MediaItem, PreviewStatus, AppSettings
from .scheduler import scheduler
from passlib.context import CryptContext

class MediaItemRead(MediaItem):
    is_bundle_hash_missing: bool = False


def priority_payload(item: MediaItem) -> dict:
    reasons = []
    if item.priority_reasons:
        try:
            parsed = json.loads(item.priority_reasons)
            if isinstance(parsed, list):
                reasons = parsed
        except (TypeError, json.JSONDecodeError):
            reasons = []

    return {
        "is_priority": item.is_priority,
        "priority_score": item.priority_score,
        "priority_reasons": reasons,
    }


# Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "change_me_in_production")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")
serializer = URLSafeTimedSerializer(SECRET_KEY)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Plex Constants
PLEX_API_PINS = "https://plex.tv/api/v2/pins"
PLEX_API_RESOURCES = "https://plex.tv/api/v2/resources"
CLIENT_ID = str(uuid.uuid4())
PRODUCT_NAME = "Plex Preview Generator"

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    scheduler.start()
    yield
    scheduler.stop()

app = FastAPI(lifespan=lifespan)

# Mount static and templates
# Ensure directories exist
os.makedirs("plex_generate_previews/web/static", exist_ok=True)
os.makedirs("plex_generate_previews/web/templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="plex_generate_previews/web/static"), name="static")
templates = Jinja2Templates(directory="plex_generate_previews/web/templates")

# Auth Utils
def create_token(username: str):
    return serializer.dumps(username, salt="login")

def get_current_user(token: Optional[str] = Cookie(None)):
    if not token:
        return None
    try:
        username = serializer.loads(token, salt="login", max_age=31536000) # 1 year (365 days)
        return username
    except:
        return None

def login_required(user: Optional[str] = Depends(get_current_user)):
    if not user:
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"}
        )
    return user

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user: Optional[str] = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login")
    
    # Check if configured
    if not scheduler.config:
        return RedirectResponse(url="/setup")

    return templates.TemplateResponse(request, "index.html", {"user": user})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")

@app.post("/login")
async def login(
    username: str = Form(...), 
    password: str = Form(...),
    session: Session = Depends(get_session)
):
    # Check AppSettings for password override
    settings = session.get(AppSettings, 1)
    is_valid = False

    # Truncate password to 72 bytes for bcrypt compatibility
    if len(password.encode('utf-8')) > 72:
        password_bytes = password.encode('utf-8')[:72]
        password = password_bytes.decode('utf-8', errors='ignore')

    try:
        if settings and settings.admin_password_hash:
            if pwd_context.verify(password, settings.admin_password_hash):
                is_valid = True
        elif password == ADMIN_PASSWORD:
            is_valid = True
    except (ValueError, Exception) as e:
        logger.error(f"Password verification error: {e}")
        is_valid = False
        
    if is_valid:
        response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        token = create_token(username)
        response.set_cookie(key="token", value=token, httponly=True, max_age=31536000)  # 1 year
        return response
    else:
        return RedirectResponse(url="/login?error=Invalid credentials", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("token")
    return response

# Settings Routes
@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, user: Optional[str] = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse(request, "settings.html", {"user": user})

@app.get("/api/settings")
async def get_settings(session: Session = Depends(get_session), user: str = Depends(login_required)):
    settings = session.get(AppSettings, 1)
    if not settings:
        return {
            "gpu_threads": 1,
            "cpu_threads": 1,
            "sync_interval": 21600,
            "mount_check_enabled": True,
            "mount_check_paths": "",
            "mount_failure_threshold": 50.0,
            "enable_multi_user_priority": False,
            "priority_history_limit": 50,
            "data_limit_gb_per_hour": 0.0,
            "rate_limit_exempt_paths": ""
        }
    return {
        "gpu_threads": settings.gpu_threads,
        "cpu_threads": settings.cpu_threads,
        "sync_interval": settings.sync_interval,
        "plex_url": settings.plex_url,
        "plex_server_name": settings.plex_server_name,
        "plex_client_identifier": settings.plex_client_identifier,
        "mount_check_enabled": settings.mount_check_enabled if hasattr(settings, 'mount_check_enabled') else True,
        "mount_check_paths": settings.mount_check_paths if hasattr(settings, 'mount_check_paths') else "",
        "mount_failure_threshold": settings.mount_failure_threshold if hasattr(settings, 'mount_failure_threshold') else 50.0,
        "enable_multi_user_priority": settings.enable_multi_user_priority if hasattr(settings, 'enable_multi_user_priority') else False,
        "priority_history_limit": settings.priority_history_limit if hasattr(settings, 'priority_history_limit') else 50,
        "data_limit_gb_per_hour": settings.data_limit_gb_per_hour if hasattr(settings, 'data_limit_gb_per_hour') else 0.0,
        "rate_limit_exempt_paths": settings.rate_limit_exempt_paths if hasattr(settings, 'rate_limit_exempt_paths') else ""
    }

@app.get("/api/settings/plex-identifier")
async def get_plex_identifier(session: Session = Depends(get_session), user: str = Depends(login_required)):
    settings = session.get(AppSettings, 1)
    if not settings or not settings.plex_client_identifier:
        raise HTTPException(status_code=404, detail="Plex client identifier not configured")
    return {"plex_client_identifier": settings.plex_client_identifier}

@app.post("/api/settings")
async def update_settings(
    payload: dict = Body(...),
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    settings = session.get(AppSettings, 1)
    if not settings:
        settings = AppSettings(id=1)
    
    # Update threads
    if "gpu_threads" in payload:
        settings.gpu_threads = int(payload["gpu_threads"])
    if "cpu_threads" in payload:
        settings.cpu_threads = int(payload["cpu_threads"])

    # Update sync interval
    if "sync_interval" in payload:
        settings.sync_interval = int(payload["sync_interval"])

    # Update mount protection settings
    if "mount_check_enabled" in payload:
        settings.mount_check_enabled = bool(payload["mount_check_enabled"])
    if "mount_check_paths" in payload:
        settings.mount_check_paths = payload["mount_check_paths"]
    if "mount_failure_threshold" in payload:
        settings.mount_failure_threshold = float(payload["mount_failure_threshold"])

    # Update multi-user priority settings
    if "enable_multi_user_priority" in payload:
        settings.enable_multi_user_priority = bool(payload["enable_multi_user_priority"])
    if "priority_history_limit" in payload:
        settings.priority_history_limit = int(payload["priority_history_limit"])

    # Update data rate limit
    if "data_limit_gb_per_hour" in payload:
        settings.data_limit_gb_per_hour = float(payload["data_limit_gb_per_hour"])
    if "rate_limit_exempt_paths" in payload:
        settings.rate_limit_exempt_paths = payload["rate_limit_exempt_paths"]

    # Update password if provided
    if "new_password" in payload and payload["new_password"]:
        password = payload["new_password"]
        # Bcrypt has a 72-byte limit - truncate if necessary
        # Using UTF-8 encoding to properly handle byte length
        if len(password.encode('utf-8')) > 72:
            # Truncate at character boundary to avoid breaking multi-byte chars
            password_bytes = password.encode('utf-8')[:72]
            # Decode back, ignoring any partial characters at the end
            password = password_bytes.decode('utf-8', errors='ignore')
            logger.warning(f"Password was truncated to 72 bytes for bcrypt compatibility")

        try:
            settings.admin_password_hash = pwd_context.hash(password)
        except ValueError as e:
            logger.error(f"Failed to hash password: {e}")
            raise HTTPException(status_code=400, detail=f"Password error: {str(e)}")
        
    session.add(settings)
    session.commit()
    
    # Reload scheduler config to pick up new thread counts
    # We can force a re-init of the worker pool by clearing it?
    # No, worker pool is complex to hot-swap. 
    # Safest is to just update the config object if possible, but pool size is fixed on init.
    # For now, we'll require a restart or implement pool resizing later.
    # Actually, let's just force a restart of the scheduler thread/pool if threads changed.
    if scheduler.config:
        if scheduler.config.gpu_threads != settings.gpu_threads or scheduler.config.cpu_threads != settings.cpu_threads:
            scheduler.config.gpu_threads = settings.gpu_threads
            scheduler.config.cpu_threads = settings.cpu_threads
            # Signal scheduler to restart pool? 
            # This is tricky without a clean restart mechanism.
            # For MVP, we just update the config object. The next time the pool is initialized (e.g. restart), it picks up.
            # But user expects immediate change.
            # Let's stop and start the scheduler?
            scheduler.stop()
            scheduler.start()

    return {"message": "Settings updated"}

@app.post("/api/settings/reset-library")
async def reset_library(
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    # Delete all media items
    statement = delete(MediaItem)
    session.exec(statement)
    session.commit()

    # Trigger resync
    scheduler.trigger_sync()

    return {"message": "Library reset. Resyncing..."}

@app.post("/api/settings/validate-mounts")
async def validate_mounts(user: str = Depends(login_required)):
    """Test mount validation - checks if configured mount paths are accessible."""
    if not scheduler.config:
        raise HTTPException(status_code=400, detail="Scheduler not configured")

    mount_valid, missing_paths = scheduler.validate_mount_paths()

    if mount_valid:
        return {
            "valid": True,
            "message": "All mount paths are accessible" if missing_paths == [] else "Mount validation passed (no paths configured)"
        }
    else:
        return {
            "valid": False,
            "message": f"Mount validation failed: {len(missing_paths)} path(s) not accessible",
            "missing_paths": missing_paths
        }

@app.post("/api/settings/reset-failed")
async def reset_failed_items(
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    """Reset all FAILED and SLOW_FAILED items back to MISSING status."""
    # Update all failed items to missing
    statement = update(MediaItem).where(
        or_(
            MediaItem.status == PreviewStatus.FAILED,
            MediaItem.status == PreviewStatus.SLOW_FAILED
        )
    ).values(
        status=PreviewStatus.MISSING,
        progress=0,
        error_message=None,
        current_processing_bundle_hash=None
    )

    result = session.exec(statement)
    session.commit()

    count = result.rowcount if hasattr(result, 'rowcount') else 0
    return {"message": f"Reset {count} failed items to MISSING status"}

@app.get("/api/settings/verify-debug")
async def verify_debug(user: str = Depends(login_required)):
    """Debug endpoint to check BIF path configuration."""
    if not scheduler.config:
        raise HTTPException(status_code=400, detail="Scheduler not configured")

    # Get a sample completed item
    with Session(engine) as session:
        sample_item = session.exec(
            select(MediaItem).where(MediaItem.status == PreviewStatus.COMPLETED).limit(1)
        ).first()

        if not sample_item:
            return {"message": "No completed items to test"}

        bif_path = None
        bif_exists = False
        parent_exists = False

        if sample_item.bundle_hash:
            bif_path = scheduler._get_bif_path(sample_item.bundle_hash)
            if bif_path:
                bif_exists = os.path.isfile(bif_path)
                parent_exists = os.path.exists(os.path.dirname(bif_path))

        return {
            "config_folder": scheduler.config.plex_config_folder,
            "sample_item": {
                "id": sample_item.id,
                "title": sample_item.title,
                "bundle_hash": sample_item.bundle_hash
            },
            "bif_path": bif_path,
            "bif_exists": bif_exists,
            "parent_dir_exists": parent_exists,
            "parent_dir": os.path.dirname(bif_path) if bif_path else None
        }

@app.post("/api/settings/discover-existing-bifs")
async def discover_existing_bifs(
    payload: dict = Body(default={}),
    user: str = Depends(login_required)
):
    """Scan MISSING items for existing BIF files and mark as COMPLETED."""
    if not scheduler.config:
        raise HTTPException(status_code=400, detail="Scheduler not configured")

    confirm = payload.get("confirm", False)
    result = scheduler.discover_existing_bifs(confirm_threshold=confirm)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    # Check if confirmation is required
    if result.get("requires_confirmation"):
        return {
            "requires_confirmation": True,
            "scanned": result["scanned"],
            "would_move": result["would_move"],
            "percentage": result["percentage"],
            "sample_paths": result.get("sample_paths", []),
            "message": f"Warning: {result['would_move']}/{result['scanned']} ({result['percentage']:.1f}%) items would be marked as COMPLETED. Confirmation required."
        }

    return {
        "message": f"Scanned {result['scanned']} MISSING items. Found {result['found']} with existing BIF files and marked as COMPLETED.",
        "scanned": result["scanned"],
        "found": result["found"]
    }

@app.post("/api/settings/verify-completed")
async def verify_completed(
    payload: dict = Body(default={}),
    user: str = Depends(login_required)
):
    """Verify that COMPLETED items have BIF files, move to MISSING if not."""
    if not scheduler.config:
        raise HTTPException(status_code=400, detail="Scheduler not configured")

    confirm = payload.get("confirm", False)
    result = scheduler.verify_completed_items(confirm_threshold=confirm)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    # Check if confirmation is required
    if result.get("requires_confirmation"):
        return {
            "requires_confirmation": True,
            "verified": result["verified"],
            "would_move": result["would_move"],
            "percentage": result["percentage"],
            "sample_paths": result.get("sample_paths", []),
            "message": f"Warning: {result['would_move']}/{result['verified']} ({result['percentage']:.1f}%) items would be moved to MISSING. Confirmation required."
        }

    return {
        "message": f"Verified {result['verified']} items. Moved {result['moved_to_missing']} to MISSING status.",
        "verified": result["verified"],
        "moved_to_missing": result["moved_to_missing"]
    }

@app.post("/api/settings/cleanup-orphaned-bundles")
async def cleanup_orphaned_bundles(
    payload: dict = Body(default={}),
    user: str = Depends(login_required)
):
    """Scan for and optionally delete orphaned bundle directories not in Plex DB."""
    if not scheduler.config:
        raise HTTPException(status_code=400, detail="Scheduler not configured")

    dry_run = payload.get("dry_run", True)
    result = scheduler.cleanup_orphaned_bundles(dry_run=dry_run)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    if dry_run:
        size_mb = result["total_size"] / 1024 / 1024
        return {
            "message": f"Dry run complete. Found {result['found']} orphaned bundles ({size_mb:.2f} MB). Run with dry_run=false to delete.",
            "found": result["found"],
            "total_size": result["total_size"],
            "dry_run": True,
            "sample_paths": result.get("sample_paths", [])
        }
    else:
        deleted_mb = result["deleted_size"] / 1024 / 1024
        return {
            "message": f"Deleted {result['deleted']} orphaned bundles (freed {deleted_mb:.2f} MB).",
            "found": result["found"],
            "deleted": result["deleted"],
            "deleted_size": result["deleted_size"],
            "dry_run": False,
            "errors": result.get("errors", [])
        }

@app.post("/api/items/{item_id}/move")
async def move_item(
    item_id: int,
    position: str = Body(..., embed=True), # "top" or "bottom" or "reset"
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    logger.debug(f"move_item endpoint hit: item_id={item_id}, position={position}")
    item = session.get(MediaItem, item_id)
    if not item:
        logger.warning(f"move_item: Item {item_id} not found.")
        raise HTTPException(status_code=404, detail="Item not found")
        
    if position == "top":
        min_order = session.exec(select(func.min(MediaItem.queue_order))).one() or 0
        item.queue_order = min_order - 1
        logger.debug(f"Moving item {item_id} to top with new queue_order: {item.queue_order}")
    elif position == "bottom":
        max_order = session.exec(select(func.max(MediaItem.queue_order))).one() or 0
        item.queue_order = max_order + 1
        logger.debug(f"Moving item {item_id} to bottom with new queue_order: {item.queue_order}")
    elif position == "reset":
        # Reset stuck item
        item.status = PreviewStatus.QUEUED
        item.progress = 0
        logger.debug(f"Resetting item {item_id} to QUEUED state.")
        
    session.add(item)
    session.commit()
    return {"message": f"Action {position} successful"}

@app.post("/api/items/{item_id}/analyze")
async def analyze_item(
    item_id: int,
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    if not scheduler.config:
        raise HTTPException(status_code=400, detail="Scheduler not configured")

    try:
        from ..plex_client import plex_server
        plex = plex_server(scheduler.config)
        plex_item = plex.fetchItem(item_id)
        plex_item.analyze()
        
        # Schedule a sync to pick up the changes
        scheduler.trigger_sync()
        
        return {"message": "Analysis triggered. Sync scheduled."}
    except Exception as e:
        logger.error(f"Analysis failed for {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/items/{item_id}/log-debug-json")
async def log_debug_json(
    item_id: int,
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    if not scheduler.config:
        raise HTTPException(status_code=400, detail="Scheduler not configured")

    try:
        from ..plex_client import plex_server
        plex = plex_server(scheduler.config)
        plex_item = plex.fetchItem(item_id)
        
        logger.info(f"--- DEBUG JSON FOR ITEM {item_id} ---")
        
        # Extract relevant media info manually to ensure we get exactly what we look for
        import json
        media_info = []
        if hasattr(plex_item, 'media'):
            for m in plex_item.media:
                parts_info = []
                if hasattr(m, 'parts'):
                    for p in m.parts:
                        parts_info.append({
                            "id": getattr(p, 'id', None),
                            "key": getattr(p, 'key', None),
                            "file": getattr(p, 'file', None),
                            "hash": getattr(p, 'hash', None),
                            "size": getattr(p, 'size', None),
                            "container": getattr(p, 'container', None),
                            "videoProfile": getattr(p, 'videoProfile', None)
                        })
                media_info.append({
                    "id": getattr(m, 'id', None),
                    "bitrate": getattr(m, 'bitrate', None),
                    "width": getattr(m, 'width', None),
                    "height": getattr(m, 'height', None),
                    "videoCodec": getattr(m, 'videoCodec', None),
                    "container": getattr(m, 'container', None),
                    "parts": parts_info
                })
        
        debug_data = {
            "ratingKey": getattr(plex_item, 'ratingKey', None),
            "title": getattr(plex_item, 'title', None),
            "type": getattr(plex_item, 'type', None),
            "guid": getattr(plex_item, 'guid', None),
            "media": media_info
        }
        
        logger.info(json.dumps(debug_data, indent=2))

        return {"message": "Debug JSON logged to server console."}
    except Exception as e:
        logger.error(f"Failed to log debug JSON for {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/items/{item_id}/force-hash-calc")
async def force_hash_calc(
    item_id: int,
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    if not scheduler.config:
        raise HTTPException(status_code=400, detail="Scheduler not configured")

    item = session.get(MediaItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found in DB")

    try:
        from ..plex_client import plex_server
        plex = plex_server(scheduler.config)
        # Validate item exists in Plex
        plex_item = plex.fetchItem(item_id)
        
        # Calculate SHA1 based on metadata key
        # Formula: SHA1("/library/metadata/<ratingKey>")
        metadata_path = f"/library/metadata/{item_id}"
        calculated_hash = hashlib.sha1(metadata_path.encode('utf-8')).hexdigest()
        
        logger.info(f"Force Hash: Calculated hash '{calculated_hash}' from metadata key '{metadata_path}'")
        
        # Update DB
        item.bundle_hash = calculated_hash
        session.add(item)
        session.commit()
        
        return {
            "message": f"Calculated hash: {calculated_hash} using metadata key: {metadata_path}",
            "hash": calculated_hash,
            "metadata_path": metadata_path
        }
    except Exception as e:
        logger.error(f"Force hash calc failed for {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/items/{item_id}/regenerate")
async def regenerate_item(
    item_id: int,
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    item = session.get(MediaItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Delete BIF if exists
    if item.bundle_hash and scheduler.config:
        bif_path = scheduler._get_bif_path(item.bundle_hash)
        if bif_path and os.path.exists(bif_path):
            try:
                os.remove(bif_path)
            except OSError as e:
                logger.error(f"Failed to delete BIF for {item_id}: {e}")
    
    # Reset status
    item.status = PreviewStatus.QUEUED
    item.progress = 0
    session.add(item)
    session.commit()
    
    scheduler.trigger_sync()
    
    return {"message": "Item queued for regeneration"}

@app.post("/api/items/{item_id}/mark-completed")
async def mark_completed_item(
    item_id: int,
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    item = session.get(MediaItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    item.status = PreviewStatus.COMPLETED
    item.progress = 100
    session.add(item)
    session.commit()

    return {"message": "Item marked as completed"}

@app.post("/api/items/{item_id}/reset-to-missing")
async def reset_to_missing(
    item_id: int,
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    item = session.get(MediaItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    item.status = PreviewStatus.MISSING
    item.progress = 0
    session.add(item)
    session.commit()

    return {"message": "Item reset to missing"}

@app.post("/api/items/{item_id}/mark-failed")
async def mark_failed_item(
    item_id: int,
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    item = session.get(MediaItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    item.status = PreviewStatus.FAILED
    item.progress = 0
    session.add(item)
    session.commit()

    return {"message": "Item marked as failed"}

@app.post("/api/items/bulk-action")
async def bulk_action(
    payload: dict = Body(...),
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    item_ids = payload.get("item_ids", [])
    action = payload.get("action")

    if not item_ids:
        raise HTTPException(status_code=400, detail="No items specified")

    if action not in ["reset", "mark_completed", "mark_failed"]:
        raise HTTPException(status_code=400, detail="Invalid action")

    updated_count = 0
    for item_id in item_ids:
        item = session.get(MediaItem, item_id)
        if item:
            if action == "reset":
                item.status = PreviewStatus.MISSING
                item.progress = 0
            elif action == "mark_completed":
                item.status = PreviewStatus.COMPLETED
                item.progress = 100
            elif action == "mark_failed":
                item.status = PreviewStatus.FAILED
                item.progress = 0
            session.add(item)
            updated_count += 1

    session.commit()

    return {"message": f"Updated {updated_count} items", "count": updated_count}

@app.get("/api/queue/status")
async def get_queue_status(session: Session = Depends(get_session), user: str = Depends(login_required)):
    # Return the paused status from both memory and database for consistency
    # Prefer database value as the source of truth
    db_paused = False
    try:
        settings = session.get(AppSettings, 1)
        if settings and hasattr(settings, 'queue_paused'):
            db_paused = settings.queue_paused
    except Exception as e:
        logger.debug(f"Could not load queue_paused from database: {e}")

    return {"paused": scheduler.paused, "db_paused": db_paused}

@app.post("/api/queue/pause")
async def pause_queue(user: str = Depends(login_required)):
    scheduler.pause()
    return {"message": "Queue paused"}

@app.post("/api/queue/resume")
async def resume_queue(user: str = Depends(login_required)):
    scheduler.resume()
    return {"message": "Queue resumed"}

# Setup Routes
@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request, user: Optional[str] = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse(request, "setup.html")

@app.post("/api/setup/plex/pin")
async def get_plex_pin(user: str = Depends(login_required)):
    headers = {
        "X-Plex-Product": PRODUCT_NAME,
        "X-Plex-Client-Identifier": CLIENT_ID,
        "Accept": "application/json"
    }
    try:
        r = requests.post(PLEX_API_PINS, headers=headers, data={"strong": "true"})
        r.raise_for_status()
        data = r.json()
        
        # Construct auth URL
        auth_url = f"https://app.plex.tv/auth#?clientID={CLIENT_ID}&code={data['code']}&context[device][product]={PRODUCT_NAME}"
        
        return {"id": data["id"], "code": data["code"], "url": auth_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/setup/plex/check/{pin_id}")
async def check_plex_pin(pin_id: int, user: str = Depends(login_required)):
    headers = {
        "X-Plex-Product": PRODUCT_NAME,
        "X-Plex-Client-Identifier": CLIENT_ID,
        "Accept": "application/json"
    }
    try:
        r = requests.get(f"{PLEX_API_PINS}/{pin_id}", headers=headers)
        r.raise_for_status()
        data = r.json()
        return {"auth_token": data.get("authToken")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/setup/plex/servers")
async def get_plex_servers(token: str, user: str = Depends(login_required)):
    headers = {
        "X-Plex-Token": token,
        "X-Plex-Client-Identifier": CLIENT_ID,
        "Accept": "application/json"
    }
    try:
        logger.info(f"Fetching Plex servers with token: {token[:5]}...{token[-5:]}")
        r = requests.get(f"{PLEX_API_RESOURCES}?includeHttps=1", headers=headers)
        r.raise_for_status()
        data = r.json()
        
        logger.debug(f"Plex Resources Response: {data}")
        
        # Log keys of first device for debugging
        if data and len(data) > 0:
            logger.debug(f"First device keys: {list(data[0].keys())}")
            logger.debug(f"First device provides: {data[0].get('provides')}")
            logger.debug(f"First device roles: {data[0].get('roles')}")

        # Filter for servers
        servers = []
        for device in data:
            roles = device.get("roles") or []
            provides_raw = device.get("provides")
            provides = provides_raw.split(",") if provides_raw else []
            
            if "server" in roles or "server" in provides:
                # Prefer local connection? Just pass all connections
                servers.append({
                    "name": device.get("name"),
                    "clientIdentifier": device.get("clientIdentifier"),
                    "accessToken": device.get("accessToken"), # Sometimes server has specific token
                    "connections": device.get("connections", [])
                })
        
        logger.info(f"Found {len(servers)} servers")
        return {"servers": servers}
    except Exception as e:
        logger.error(f"Error fetching servers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/setup/save")
async def save_config(
    payload: dict = Body(...),
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    plex_url = payload.get("plex_url")
    plex_token = payload.get("plex_token")
    plex_server_name = payload.get("plex_server_name")
    plex_client_identifier = payload.get("plex_client_identifier")
    
    if not plex_url or not plex_token or not plex_client_identifier:
        raise HTTPException(status_code=400, detail="Missing URL, Token or Client Identifier")

    # Save to DB
    settings = session.get(AppSettings, 1)
    if not settings:
        settings = AppSettings(id=1)
    
    settings.plex_url = plex_url
    settings.plex_token = plex_token
    settings.plex_server_name = plex_server_name
    settings.plex_client_identifier = plex_client_identifier
    session.add(settings)
    session.commit()
    
    # Reload scheduler configuration immediately
    logger.info("Setup saved, reloading scheduler configuration")
    scheduler._try_load_config()

    # Start scheduler if it wasn't running
    if scheduler.stop_event.is_set():
        logger.info("Starting scheduler after setup")
        scheduler.start()

    return {"message": "Saved"}

# API Endpoints
@app.get("/api/stats")
async def get_stats(session: Session = Depends(get_session), user: str = Depends(login_required)):
    # Expire all cached objects to ensure fresh data from database
    session.expire_all()

    total = session.exec(select(func.count(MediaItem.id))).one()
    completed = session.exec(select(func.count(MediaItem.id)).where(MediaItem.status == PreviewStatus.COMPLETED)).one()
    queued = session.exec(select(func.count(MediaItem.id)).where(MediaItem.status == PreviewStatus.QUEUED)).one()
    missing = session.exec(select(func.count(MediaItem.id)).where(MediaItem.status == PreviewStatus.MISSING)).one()
    processing = session.exec(select(func.count(MediaItem.id)).where(MediaItem.status == PreviewStatus.PROCESSING)).one()
    priority = session.exec(select(func.count(MediaItem.id)).where(MediaItem.is_priority == True)).one()
    failed = session.exec(select(func.count(MediaItem.id)).where(
        col(MediaItem.status).in_([PreviewStatus.FAILED, PreviewStatus.SLOW_FAILED])
    )).one()
    media_missing = session.exec(select(func.count(MediaItem.id)).where(MediaItem.status == PreviewStatus.MEDIA_MISSING)).one()

    # Get last sync time and summary from AppSettings
    settings = session.get(AppSettings, 1)
    last_sync_time = settings.last_sync_time if settings else None
    last_sync_summary = settings.last_sync_summary if settings else None

    # Get rate limit info
    limit_gb = settings.data_limit_gb_per_hour if settings else 0.0
    processed_bytes = settings.total_bytes_processed_hour if settings else 0
    limit_bytes = int(limit_gb * 1024 * 1024 * 1024) if limit_gb > 0 else 0
    remaining_bytes = limit_bytes - processed_bytes if limit_gb > 0 else 0

    rate_limit_info = {
        "limit_gb": limit_gb,
        "is_limited": scheduler._rate_limited,
        "processed_bytes": processed_bytes,
        "remaining_bytes": remaining_bytes,
        "message": scheduler._rate_limit_message if scheduler._rate_limited else ""
    }
    
    # Ensure is_limited is up to date
    if rate_limit_info["limit_gb"] > 0:
        rate_limit_info["is_limited"] = scheduler._check_rate_limit()
        rate_limit_info["message"] = scheduler._rate_limit_message if scheduler._rate_limited else ""

    return {
        "total": total,
        "completed": completed,
        "queued": queued,
        "missing": missing,
        "processing": processing,
        "priority": priority,
        "failed": failed,
        "media_missing": media_missing,
        "last_sync_time": (last_sync_time.isoformat() + "Z") if last_sync_time else None,
        "last_sync_summary": last_sync_summary,
        "rate_limit": rate_limit_info,
        "sync_in_progress": scheduler.sync_in_progress
    }

@app.get("/api/libraries")
async def get_libraries(session: Session = Depends(get_session), user: str = Depends(login_required)):
    libraries = session.exec(select(MediaItem.library_name).distinct()).all()
    return sorted(libraries)

@app.get("/api/items")
async def get_items(
    status: Optional[str] = None,
    search: Optional[str] = None,
    library_name: Optional[str] = None,
    hide_completed: bool = False, # Filter to hide completed items
    show_priority_only: bool = False, # Filter to show only priority items
    sort_by: Optional[str] = None, # Column to sort by: added_at, title, avg_speed, status
    sort_order: Optional[str] = "desc", # Sort order: asc or desc
    page: int = 1,
    limit: int = 50,
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    # Base query
    filters = []
    if status:
        filters.append(MediaItem.status == status)
    if search:
        filters.append(col(MediaItem.title).contains(search))
    if library_name: # Apply library filter
        filters.append(MediaItem.library_name == library_name)
    if hide_completed: # Exclude completed items
        filters.append(MediaItem.status != PreviewStatus.COMPLETED)
    if show_priority_only: # Show only priority items
        filters.append(MediaItem.is_priority == True)
    
    # Count total matching
    count_query = select(func.count()).select_from(MediaItem)
    if filters:
        count_query = count_query.where(*filters)
    total = session.exec(count_query).one()
    
    # Get items
    query = select(MediaItem)
    if filters:
        query = query.where(*filters)
    
    # Sort
    if sort_by:
        # User-selected sorting
        sort_column = None
        if sort_by == "added_at":
            sort_column = MediaItem.added_at
        elif sort_by == "title":
            sort_column = MediaItem.title
        elif sort_by == "status":
            sort_column = MediaItem.status
        elif sort_by == "avg_speed":
            # For avg_speed, we need to handle it specially since it's stored as string (e.g., "1.23x")
            # Use SQLite's CAST to convert to numeric for sorting, handling NULL values
            # REPLACE removes 'x' suffix, NULLIF handles empty strings
            sort_column = case(
                (MediaItem.avg_speed.is_(None), None),
                else_=func.cast(func.replace(MediaItem.avg_speed, 'x', ''), Float)
            )

        if sort_column is not None:
            if sort_order == "asc":
                # Put NULLs last for ascending order
                query = query.order_by(sort_column.asc().nullslast())
            else:
                # Put NULLs last for descending order
                query = query.order_by(sort_column.desc().nullslast())
    else:
        # Default sort mirrors scheduler processing order so the list shows what will run next.
        query = query.order_by(
            case(
                (
                    col(MediaItem.status).in_(
                        [PreviewStatus.PROCESSING, PreviewStatus.QUEUED, PreviewStatus.MISSING]
                    ),
                    0,
                ),
                else_=1,
            ),
            MediaItem.priority_score.desc(),
            MediaItem.updated_at.desc(),
            MediaItem.queue_order.asc(),
        )
    
    # Pagination
    offset = (page - 1) * limit
    query = query.offset(offset).limit(limit)
    
    items = session.exec(query).all()

    # Expand items to show one row per media part
    expanded_items = []
    for item in items:
        if item.media_parts_info:
            try:
                import json
                parts = json.loads(item.media_parts_info)
                for idx, part in enumerate(parts):
                    # Determine part status based on processing state and BIF existence
                    bif_exists = part.get('bif_path') and os.path.exists(part.get('bif_path'))
                    part_bundle_hash = part.get('bundle_hash')
                    
                    # Check if this specific part is the one being processed
                    is_processing_this_part = (
                        item.current_processing_bundle_hash == part_bundle_hash 
                        if item.current_processing_bundle_hash and part_bundle_hash else False
                    )

                    part_progress = 0
                    part_status = PreviewStatus.MISSING
                    part_error = None
                    part_speed = None

                    if is_processing_this_part and item.status in [PreviewStatus.PROCESSING, PreviewStatus.FAILED, PreviewStatus.SLOW_FAILED]:
                        # This part is currently processing (or just failed)
                        part_status = item.status
                        part_progress = item.progress
                        part_error = item.error_message if item.status in [PreviewStatus.FAILED, PreviewStatus.SLOW_FAILED] else None
                        part_speed = item.avg_speed
                    elif bif_exists:
                        # Part is completed
                        part_status = PreviewStatus.COMPLETED
                        part_progress = 100
                        part_speed = item.avg_speed  # Preserve avg speed for completed items
                    else:
                        # Part is pending or missing
                        if item.status == PreviewStatus.QUEUED:
                            part_status = PreviewStatus.QUEUED
                        elif item.status == PreviewStatus.MEDIA_MISSING:
                            part_status = PreviewStatus.MEDIA_MISSING
                        elif item.status in [PreviewStatus.FAILED, PreviewStatus.SLOW_FAILED]:
                            part_status = item.status
                            part_error = item.error_message
                            part_speed = item.avg_speed
                        else:
                            part_status = PreviewStatus.MISSING

                    # Extract filename from path for part label
                    file_path = part.get('file_path', '')
                    filename = os.path.basename(file_path) if file_path else f"Part {idx + 1}"

                    # Check for symlink
                    symlink_target = None
                    if file_path and os.path.islink(file_path):
                        try:
                            symlink_target = os.readlink(file_path)
                        except OSError:
                            pass

                    expanded_items.append({
                        "id": f"{item.id}_part{idx}",  # Unique ID per part
                        "item_id": item.id,  # Original item ID
                        "part_index": idx,
                        "title": f"{item.title} - {filename}",
                        "media_type": item.media_type,
                        "library_name": item.library_name,
                        "file_path": part.get('file_path'),
                        "symlink_target": symlink_target,
                        "bundle_hash": part.get('bundle_hash'),
                        "bif_path": part.get('bif_path'),
                        "status": part_status,
                        "progress": part_progress,
                        "avg_speed": part_speed,
                        "added_at": item.added_at,
                        "updated_at": item.updated_at,
                        "error_message": part_error,
                        "is_multi_part": len(parts) > 1,
                        **priority_payload(item),
                    })
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Error parsing media_parts_info for item {item.id}: {e}")
                # Fallback to single row if parsing fails
                expanded_items.append({
                    "id": item.id,
                    "item_id": item.id,
                    "part_index": 0,
                    "title": item.title,
                    "media_type": item.media_type,
                    "library_name": item.library_name,
                    "file_path": item.file_path,
                    "bundle_hash": item.bundle_hash,
                    "bif_path": item.bif_path,
                    "status": item.status,
                    "progress": item.progress,
                    "avg_speed": item.avg_speed,
                    "added_at": item.added_at,
                    "updated_at": item.updated_at,
                    "error_message": item.error_message,
                    "is_multi_part": False,
                    **priority_payload(item),
                })
        else:
            # No parts info, use item directly
            symlink_target = None
            if item.file_path and os.path.islink(item.file_path):
                try:
                    symlink_target = os.readlink(item.file_path)
                except OSError:
                    pass

            expanded_items.append({
                "id": item.id,
                "item_id": item.id,
                "part_index": 0,
                "title": item.title,
                "media_type": item.media_type,
                "library_name": item.library_name,
                "file_path": item.file_path,
                "symlink_target": symlink_target,
                "bundle_hash": item.bundle_hash,
                "bif_path": item.bif_path,
                "status": item.status,
                "progress": item.progress,
                "avg_speed": item.avg_speed,
                "added_at": item.added_at,
                "updated_at": item.updated_at,
                "error_message": item.error_message,
                "is_multi_part": False,
                **priority_payload(item),
            })

    return {
        "items": expanded_items,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": (total + limit - 1) // limit if limit > 0 else 1
    }

@app.get("/api/items/processing")
async def get_processing_items(
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    """Get items currently being processed or queued (for the processing section on home page)."""
    import json as json_module

    # Get processing items first, then queued items
    processing_query = select(MediaItem).where(
        col(MediaItem.status).in_([PreviewStatus.PROCESSING, PreviewStatus.QUEUED])
    ).order_by(
        # Processing items first, then queued by priority and queue order
        case(
            (MediaItem.status == PreviewStatus.PROCESSING, 0),
            else_=1
        ),
        MediaItem.priority_score.desc(),
        MediaItem.queue_order.asc()
    ).limit(10)  # Limit to 10 items for the UI

    items = session.exec(processing_query).all()

    result_items = []
    for item in items:
        # Get file path from media_parts_info if available
        file_path = item.file_path
        if item.media_parts_info:
            try:
                parts = json_module.loads(item.media_parts_info)
                if parts and len(parts) > 0:
                    file_path = parts[0].get('file_path', item.file_path)
            except (json_module.JSONDecodeError, KeyError):
                pass

        # Resolve symlink target if file is a symlink
        symlink_target = None
        if file_path:
            try:
                if os.path.islink(file_path):
                    symlink_target = os.readlink(file_path)
            except (OSError, IOError):
                pass

        result_items.append({
            "id": item.id,
            "title": item.title,
            "media_type": item.media_type,
            "library_name": item.library_name,
            "status": item.status,
            "progress": item.progress,
            "avg_speed": item.avg_speed,
            **priority_payload(item),
            "file_path": file_path,
            "symlink_target": symlink_target
        })

    return {"items": result_items}

@app.get("/api/items/all-with-hash-status")
async def get_items_all_with_hash_status(
    search: Optional[str] = None,
    page: int = 1,
    limit: int = 20,
    session: Session = Depends(get_session),
    user: str = Depends(login_required)
):
    # Base query for all MediaItems
    filters = []
    
    # Apply search filter if present (title OR bundle_hash)
    if search:
        filters.append(
            or_(
                col(MediaItem.title).contains(search),
                col(MediaItem.bundle_hash).contains(search)
            )
        )
    
    # Count total matching items before pagination
    count_query = select(func.count()).select_from(MediaItem)
    if filters:
        count_query = count_query.where(*filters)
    total = session.exec(count_query).one()


    # Get items with pagination and sorting
    query = select(MediaItem)
    if filters:
        query = query.where(*filters)
    query = query.order_by(MediaItem.id) # Sort by ID for stable pagination
    offset = (page - 1) * limit
    items_from_db = session.exec(query.offset(offset).limit(limit)).all()

    # Convert to MediaItemRead and add is_bundle_hash_missing flag
    items_with_status = []
    for item in items_from_db:
        item_read = MediaItemRead.model_validate(item)
        item_read.is_bundle_hash_missing = item.bundle_hash is None or item.bundle_hash == ""
        items_with_status.append(item_read)

    return {
        "items": items_with_status,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": (total + limit - 1) // limit if limit > 0 else 1
    }

@app.post("/api/sync")
async def trigger_sync(user: str = Depends(login_required)):
    scheduler.trigger_sync()
    return {"message": "Sync triggered (scheduled)"}


def start():
    import uvicorn
    uvicorn.run("plex_generate_previews.web.main:app", host="0.0.0.0", port=8008, reload=False, access_log=False)
