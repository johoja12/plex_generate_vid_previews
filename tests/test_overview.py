from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine, select
from sqlmodel.pool import StaticPool
from plex_generate_previews.web.models import AppSettings, MediaItem, MediaType, PreviewStatus, PreviewCompletion
from plex_generate_previews.web.overview import overview_summary
from plex_generate_previews.web import main, scheduler as scheduler_module


@pytest.fixture
def db():
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def item(id, **kwargs):
    return MediaItem(id=id, title=f'Episode {id}', media_type=MediaType.EPISODE, library_name='TV', **kwargs)


def test_hourly_count_is_unique_and_ignores_metadata_timestamps(db):
    now = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)
    db.add(item(1, status=PreviewStatus.COMPLETED, updated_at=now))
    db.add(AppSettings(id=1))
    for key, minutes in [(1, 5), (1, 15), (2, 60), (3, 61), (4, -1)]:
        db.add(PreviewCompletion(media_item_id=key, title=f'Item {key}', completed_at=now-timedelta(minutes=minutes)))
    db.commit()
    summary = overview_summary(db, now)
    assert summary['completed_hour'] == 2
    assert sum(summary['hour_bins']) == 2
    assert summary['recent_completed'][0]['title'] == 'Item 1'
    db.get(MediaItem, 1).updated_at = now + timedelta(minutes=1)
    db.commit()
    assert overview_summary(db, now)['completed_hour'] == 2


def test_completion_events_are_written_once_and_survive_database_reopen(tmp_path, monkeypatch):
    path = tmp_path/'events.db'
    engine = create_engine(f'sqlite:///{path}')
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(scheduler_module, 'engine', engine)
    monkeypatch.setattr(scheduler_module.scheduler, 'config', SimpleNamespace(plex_config_folder='/tmp', preview_part_scopes={}))
    with Session(engine) as session:
        session.add(item(1, status=PreviewStatus.PROCESSING)); session.commit()
    manager = scheduler_module.DbProgressManager()
    manager._write_update_immediate('1', {'progress_percent': 100})
    manager._write_update_immediate('1', {'progress_percent': 100})
    engine.dispose()
    reopened = create_engine(f'sqlite:///{path}')
    with Session(reopened) as session:
        events = session.exec(select(PreviewCompletion)).all()
        assert len(events) == 1
        assert session.get(MediaItem, 1).status == PreviewStatus.COMPLETED
    reopened.dispose()


def test_failed_update_and_manual_completion_do_not_count_as_generated(db, monkeypatch):
    from plex_generate_previews.web.overview import record_completion
    media = item(1, status=PreviewStatus.FAILED)
    record_completion(db, media, PreviewStatus.PROCESSING, {'progress_percent': 100, 'failed': True})
    db.add(media); db.commit()
    assert db.exec(select(PreviewCompletion)).all() == []


def test_overview_matches_priority_scope_and_deduplicates_sources(db):
    db.add(AppSettings(id=1, full_backlog=False))
    db.add(item(1, is_priority=True, priority_reasons='[{"type":"hub","hub":"Favourite TV"},{"type":"hub","hub":"Favourite TV"},{"type":"next_episode"}]'))
    db.add(item(2, status=PreviewStatus.PROCESSING, is_priority=True, priority_reasons='[{"type":"hub","hub":"Favourite TV"}]'))
    db.add(item(3))
    db.add(item(4, is_priority=True, priority_reasons='invalid'))
    db.commit()
    summary = overview_summary(db)
    assert summary['pending'] == 2
    assert summary['sources'] == {'hub:Favourite TV': 2, 'type:next_episode': 1}
    db.get(AppSettings, 1).full_backlog = True; db.commit()
    assert overview_summary(db)['pending'] == 3


def test_origin_filters_work_before_pagination_and_stats_require_auth(db):
    db.add(AppSettings(id=1))
    for key in range(1, 5):
        db.add(item(key, is_priority=True, priority_reasons='[{"type":"hub","hub":"Favourite TV"}]' if key%2 else '[{"type":"next_episode"}]'))
    db.commit()
    main.app.dependency_overrides[main.get_session] = lambda: db
    try:
        client = TestClient(main.app)
        assert client.get('/api/overview', follow_redirects=False).status_code == 307
        client.cookies.set('token', main.create_token('admin'))
        result = client.get('/api/items', params={'origin':'hub:Favourite TV','limit':1,'pending_only':'true'}).json()
        assert result['total'] == 2 and len(result['items']) == 1
        result = client.get('/api/items', params={'origin':'type:next_episode','pending_only':'true','media_type':'episode'}).json()
        assert result['total'] == 2
        assert client.get('/api/overview').json()['pending'] == 4
        settings = client.get('/api/settings').json()
        assert settings['use_database_sync'] is True
        assert 'plex_token' not in settings and 'admin_password_hash' not in settings
    finally:
        main.app.dependency_overrides.clear()
