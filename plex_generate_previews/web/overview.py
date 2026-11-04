"""Dashboard summaries and durable generated-preview completion events."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from sqlmodel import select, func, col
from .models import AppSettings, MediaItem, PreviewCompletion, PreviewStatus


def record_completion(session, item, previous_status, data):
    if (data.get('progress_percent', 0) >= 100 and not data.get('failed')
            and previous_status != PreviewStatus.COMPLETED
            and item.status == PreviewStatus.COMPLETED):
        session.add(PreviewCompletion(media_item_id=item.id, title=item.title))


def overview_summary(session, now=None):
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=1)
    counts = {status.name: count for status, count in session.exec(
        select(MediaItem.status, func.count()).group_by(MediaItem.status)).all()}
    settings = session.get(AppSettings, 1)
    eligible = select(MediaItem).where(col(MediaItem.status).in_(
        [PreviewStatus.MISSING, PreviewStatus.QUEUED, PreviewStatus.PROCESSING]))
    priority_items = session.exec(eligible.where(MediaItem.is_priority == True)).all()
    sources = Counter()
    for item in priority_items:
        try:
            reasons = json.loads(item.priority_reasons or '[]')
            if not isinstance(reasons, list):
                continue
            sources.update(set(
                ('hub:' + str(reason.get('hub', 'Plex hub'))) if reason.get('type') == 'hub'
                else ('type:' + str(reason.get('type', 'priority')))
                for reason in reasons if isinstance(reason, dict)))
        except (ValueError, TypeError):
            continue
    events = session.exec(select(PreviewCompletion).where(
        PreviewCompletion.completed_at >= cutoff,
        PreviewCompletion.completed_at <= now).order_by(PreviewCompletion.completed_at.desc())).all()
    unique = {}
    for event in events:
        unique.setdefault(event.media_item_id, event)
    bins = [0] * 6
    for event in unique.values():
        bins[min(5, int((event.completed_at - cutoff).total_seconds() // 600))] += 1
    backlog = bool(settings and settings.full_backlog)
    pending = ((counts.get('MISSING', 0) + counts.get('QUEUED', 0)) if backlog else
               sum(item.status != PreviewStatus.PROCESSING for item in priority_items))
    return {
        'counts': counts, 'pending': pending, 'sources': dict(sources),
        'completed_hour': len(unique), 'hour_bins': bins,
        'recent_completed': [{'title': event.title, 'at': event.completed_at.isoformat() + 'Z'}
                             for event in list(unique.values())[:8]],
        'snapshot_at': now.isoformat() + 'Z',
        'settings': {'queue_paused': bool(settings and settings.queue_paused), 'full_backlog': backlog,
                     'last_sync_time': settings.last_sync_time.isoformat() + 'Z' if settings and settings.last_sync_time else None},
    }


def library_connection_health(config):
    """Use the configured data source, including legacy SQLite installations."""
    import importlib
    import time
    try:
        database = importlib.import_module('plex_generate_previews.plex_database')
    except ModuleNotFoundError as error:
        if error.name != 'plex_generate_previews.plex_database':
            raise
        from ..plex_client import open_plex_db
        started = time.monotonic()
        connection = open_plex_db(config.plex_config_folder)
        try:
            connection.execute('SELECT id FROM metadata_items LIMIT 1').fetchone()
        finally:
            connection.close()
        return {'backend': 'sqlite', 'healthy': True, 'latency_ms': round((time.monotonic()-started)*1000, 1)}
    return {'backend': getattr(config, 'plex_db_backend', 'sqlite'), **database.get_data_source(config).health()}
