import asyncio
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests
from sqlalchemy import text
from sqlmodel import SQLModel, Session, create_engine, select
from sqlmodel.pool import StaticPool

from plex_generate_previews.web import database, infini_cache, main, scheduler as scheduler_module
from plex_generate_previews.web.models import AppSettings, MediaItem, MediaType, PreviewStatus
from plex_generate_previews.web.scheduler import Scheduler
from plex_generate_previews.web.priority import PriorityInfo, set_playback_priority


@pytest.fixture
def db(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(scheduler_module, "engine", engine)
    yield engine
    engine.dispose()


def movie(key, **kwargs):
    return MediaItem(id=key, title=f"Movie {key}", library_name="Movies", media_type=MediaType.MOVIE, **kwargs)


def test_scope_filters_initial_and_refill_and_stops_on_setting_change(db, monkeypatch):
    with Session(db) as session:
        session.add(AppSettings(id=1, full_backlog=False))
        session.add(movie(1))
        session.add(movie(2, is_priority=True, priority_score=999))
        session.add(movie(3, status=PreviewStatus.QUEUED))
        session.commit()
    scheduler = Scheduler()
    scheduler.config = SimpleNamespace(gpu_threads=1, cpu_threads=0)
    scheduler.worker_pool = MagicMock()
    monkeypatch.setattr(scheduler, "validate_mount_paths", lambda: (True, []))
    monkeypatch.setattr(scheduler, "_check_rate_limit", lambda: False)
    monkeypatch.setattr(scheduler_module, "plex_server", lambda config: MagicMock())

    def process(items, *args, **kwargs):
        assert [int(row[0]) for row in items] == [2]
        with Session(db) as session:
            item = session.get(MediaItem, 2)
            item.status = PreviewStatus.PROCESSING
            session.add(item)
            session.commit()
        assert kwargs["fetch_more_items"]() == []  # No leakage of ordinary QUEUED work
        with Session(db) as session:
            settings = session.get(AppSettings, 1)
            settings.full_backlog = True
            session.add(settings)
            session.commit()
        assert kwargs["stop_condition"]()
        assert kwargs["fetch_more_items"]()  # Full backlog restores remaining work

    scheduler.worker_pool.process_items.side_effect = process
    scheduler.process_queue()
    with Session(db) as session:
        assert session.get(MediaItem, 1).status in (PreviewStatus.MISSING, PreviewStatus.QUEUED)
        assert len(session.exec(select(MediaItem)).all()) == 3


def test_settings_round_trip_and_pending_ui_scope(db):
    with Session(db) as session:
        assert asyncio.run(main.get_settings(session=session, user="test"))["full_backlog"] is False
        session.add(movie(1))
        session.add(movie(2, is_priority=True))
        session.commit()
        limited = asyncio.run(main.get_items(session=session, user="test"))
        assert limited["total"] == 1
        stats = asyncio.run(main.get_stats(session=session, user="test"))
        assert stats["missing"] == 1
        asyncio.run(main.update_settings({"full_backlog": True}, session=session, user="test"))
        assert asyncio.run(main.get_settings(session=session, user="test"))["full_backlog"] is True
        assert asyncio.run(main.get_items(session=session, user="test"))["total"] == 2
        with pytest.raises(main.HTTPException):
            asyncio.run(main.update_settings({"full_backlog": "false"}, session=session, user="test"))


@pytest.mark.parametrize("full_backlog", [False, True])
def test_queue_orders_playback_then_cached_then_up_next(db, full_backlog):
    with Session(db) as session:
        session.add(AppSettings(id=1, full_backlog=full_backlog))
        session.add(movie(1, is_priority=True, priority_score=999, status=PreviewStatus.QUEUED))
        session.add(movie(2, is_priority=True, priority_score=1999, status=PreviewStatus.QUEUED))
        session.add(movie(3, is_priority=True, priority_score=4000))
        session.commit()
        scheduler = Scheduler()
        assert [item.id for item in session.exec(scheduler._queue_candidates(session, 10)).all()] == [3, 2, 1]


def test_priority_refresh_interval_accounts_for_scan_runtime(db, monkeypatch):
    scheduler = Scheduler()
    scheduler.config = SimpleNamespace()
    scheduler.last_sync_time = datetime.utcnow()
    scheduler.stop_event = MagicMock()
    scheduler.stop_event.is_set.side_effect = [False, True]
    scheduler.sync_wake_event = MagicMock()
    scheduler.refresh_priority_metadata = MagicMock()
    scheduler.sync_library = MagicMock()
    monkeypatch.setattr(scheduler_module, "time", SimpleNamespace(monotonic=MagicMock(side_effect=[100, 145])))
    scheduler._sync_loop()
    scheduler.refresh_priority_metadata.assert_called_once()
    scheduler.sync_library.assert_not_called()
    scheduler.sync_wake_event.wait.assert_called_once_with(15)


def test_existing_settings_migrate_to_priority_only(db, monkeypatch):
    with Session(db) as session:
        session.add(AppSettings(id=1, full_backlog=True))
        session.commit()
    with db.connect() as connection:
        connection.execute(text("ALTER TABLE appsettings DROP COLUMN full_backlog"))
        connection.commit()
    monkeypatch.setattr(database, "engine", db)
    database.create_db_and_tables()
    database.create_db_and_tables()  # Idempotent
    with Session(db) as session:
        assert session.get(AppSettings, 1).full_backlog is False


def test_active_playback_selects_video_sessions_only():
    plex = MagicMock()
    plex.sessions.return_value = [
        SimpleNamespace(ratingKey=str(i), type=kind, players=[SimpleNamespace(state=state)])
        for i, kind, state in [(1, "movie", "playing"), (2, "episode", "paused"),
                               (3, "episode", "buffering"), (4, "track", "playing"),
                               (5, "movie", "stopped")]
    ]
    assert Scheduler()._collect_active_playback_rating_keys(plex) == {1, 2, 3}


def test_detection_combines_playback_and_cache_with_existing_sources(db, monkeypatch):
    scheduler = Scheduler()
    with Session(db) as session:
        session.add(movie(2, file_path="/cached"))
        session.commit()
    monkeypatch.setattr(scheduler, "_read_priority_watch_events", lambda limit: [])
    monkeypatch.setattr(scheduler, "_read_recent_partial_rating_keys", lambda now: {3})
    monkeypatch.setattr(scheduler, "_read_priority_episode_rows", lambda: [])
    monkeypatch.setattr(scheduler, "_collect_priority_hub_items", lambda plex: {})
    monkeypatch.setattr(scheduler, "_collect_on_deck_rating_keys", lambda plex: {3})
    monkeypatch.setattr(scheduler, "_collect_active_playback_rating_keys", lambda plex: {1, 99})
    monkeypatch.setattr(scheduler_module, "fully_cached_paths", lambda candidates=None: {"/cached"})
    result = scheduler.detect_priority_items(MagicMock(), {1, 2, 3})
    assert set(result) == {1, 2, 3}
    assert result[1].score > result[3].score > result[2].score
    assert result[2].reasons[0]["type"] == "fully_cached"


def test_playback_tier_preserves_cache_when_session_ends():
    info = PriorityInfo(reasons=[{"type": "on_deck", "score": 900}, {"type": "next_episode", "score": 800}, {"type": "fully_cached", "score": 1000}])
    set_playback_priority(info, True)
    assert info.score == 5999
    set_playback_priority(info, True)
    assert len([reason for reason in info.reasons if reason["type"] == "active_playback"]) == 1
    set_playback_priority(info, False)
    assert info.score == 1999
    assert {reason["type"] for reason in info.reasons} == {"on_deck", "next_episode", "fully_cached"}


def test_playback_api_includes_completed_and_unknown_items_and_caches_plex(db, monkeypatch):
    scheduler = Scheduler()
    plex = MagicMock()
    plex.sessions.return_value = [SimpleNamespace(
        ratingKey=str(key), type="episode", title="Episode", grandparentTitle="A Show",
        players=[SimpleNamespace(state="playing", title="Living room")],
        usernames=["Viewer"], sessionKey=key, duration=3600000, viewOffset=120000,
    ) for key in [1, 2]]
    snapshot = scheduler.get_active_playback(plex)
    scheduler.get_active_playback(plex)
    plex.sessions.assert_called_once()
    monkeypatch.setattr(main.scheduler, "get_active_playback", lambda: snapshot)
    with Session(db) as session:
        session.add(movie(1, status=PreviewStatus.COMPLETED, progress=100))
        session.commit()
        result = asyncio.run(main.get_playback(session=session, user="test"))
    assert result["available"] is True
    assert len(result["sessions"]) == 2
    assert result["sessions"][0]["title"] == "A Show — Episode"
    assert result["sessions"][0]["viewer"] == "Viewer"
    assert result["sessions"][0]["device"] == "Living room"
    assert result["sessions"][0]["preview_status"] == "completed"
    assert result["sessions"][1]["preview_status"] is None


def test_playback_refresh_promotes_and_demotes_without_dropping_other_sources(db, monkeypatch):
    scheduler = Scheduler()
    with Session(db) as session:
        session.add(movie(1, is_priority=True, priority_score=1000,
                          priority_reasons='[{"type":"fully_cached","score":1000}]'))
        session.add(movie(2, is_priority=True, priority_score=2000,
                          priority_reasons='[{"type":"active_playback","score":2000}]'))
        session.commit()
    monkeypatch.setattr(scheduler, "get_active_playback", lambda: {"available": True, "sessions": [{"rating_key": 1}]})
    monkeypatch.setattr(scheduler, "_read_priority_episode_rows", lambda: [])
    scheduler.refresh_active_playback_metadata()
    with Session(db) as session:
        assert session.get(MediaItem, 1).priority_score == 5000
        assert session.get(MediaItem, 2).priority_score == 0
        assert session.get(MediaItem, 2).is_priority is False
    monkeypatch.setattr(scheduler, "get_active_playback", lambda: {"available": False, "sessions": []})
    scheduler.refresh_active_playback_metadata()
    with Session(db) as session:
        assert session.get(MediaItem, 1).priority_score == 5000


def test_cache_catalogue_paginates_and_requires_verified_current_size(monkeypatch):
    monkeypatch.setenv("INFINIDYSK_URL", "http://infini")
    monkeypatch.setenv("INFINIDYSK_API_KEY", "test-key")
    ids = [f"00000000-0000-0000-0000-{i:012d}" for i in range(1, 5)]
    responses = [
        {"folders": [{"id": "online", "online": True}, {"id": "offline", "online": False}]},
        {"entries": [{"itemId": ids[0], "length": 100, "verifiedBytes": 100},
                     {"itemId": ids[1], "length": 100, "verifiedBytes": 99, "allocatedBytes": 100},
                     {"itemId": ids[2], "length": 0, "verifiedBytes": 0}], "nextAfter": "cursor"},
        {"entries": [{"itemId": ids[3], "length": 100, "verifiedBytes": 100}], "nextAfter": None},
        {"size": 100, "contentPath": "/content/file.mkv",
         "mappings": [{"status": "valid", "linkPath": "/mnt/plex/movie.mkv"}]},
        {"size": 101, "contentPath": "/content/stale.mkv"},
    ]
    client = MagicMock()
    client.__enter__.return_value = client
    client.get.side_effect = [MagicMock(json=lambda value=value: value) for value in responses]
    monkeypatch.setattr(infini_cache.requests, "Session", lambda: client)
    paths = infini_cache.fully_cached_paths()
    assert paths == {f"/mnt/remote/infinidysk/.ids/{ids[0]}",
                     "/mnt/remote/infinidysk/content/file.mkv", "/mnt/plex/movie.mkv"}
    assert client.get.call_args_list[2].kwargs["params"]["after"] == "cursor"
    client.get.side_effect = requests.ConnectionError("private error")
    assert infini_cache.fully_cached_paths() == set()


def test_cache_item_accepts_one_cached_version_and_matches_symlink_uuid(tmp_path):
    key = "00000000-0000-0000-0000-000000000001"
    link = tmp_path / "movie.mkv"
    link.symlink_to("/mnt/remote/infinidysk/.ids/" + key.replace("-", "") + ".mkv")
    item = movie(1, file_path=str(link))
    cached = {"/mnt/remote/infinidysk/.ids/" + key}
    assert infini_cache.item_is_fully_cached(item, cached)
    item.media_parts_info = json.dumps([{"file_path": str(link)}, {"file_path": "/cold"}])
    assert infini_cache.item_is_fully_cached(item, cached)
    assert infini_cache.cached_part_paths(item, cached) == {str(link)}
    item.media_parts_info = "invalid"
    assert not infini_cache.item_is_fully_cached(item, cached)


def test_deleted_cached_item_does_not_hide_other_fully_cached_files(monkeypatch):
    monkeypatch.setenv("INFINIDYSK_URL", "http://infini")
    monkeypatch.setenv("INFINIDYSK_API_KEY", "test-key")
    ids = [f"00000000-0000-0000-0000-{i:012d}" for i in (1, 2)]
    response = lambda payload: MagicMock(status_code=200, json=lambda: payload)
    client = MagicMock()
    client.__enter__.return_value = client
    client.get.side_effect = [
        response({"folders": [{"id": "online", "online": True}]}),
        response({"entries": [{"itemId": key, "length": 100, "verifiedBytes": 100} for key in ids], "nextAfter": None}),
        MagicMock(status_code=404),
        response({"size": 100, "contentPath": "/content/valid.mkv"}),
    ]
    monkeypatch.setattr(infini_cache.requests, "Session", lambda: client)
    paths = infini_cache.fully_cached_paths()
    assert paths == {f"/mnt/remote/infinidysk/.ids/{ids[1]}", "/mnt/remote/infinidysk/content/valid.mkv"}


def test_relative_infini_library_mapping_uses_library_root(monkeypatch):
    monkeypatch.setenv("INFINIDYSK_URL", "http://infini")
    monkeypatch.setenv("INFINIDYSK_API_KEY", "test-key")
    monkeypatch.setenv("INFINIDYSK_LIBRARY_PATH", "/library")
    key = "00000000-0000-0000-0000-000000000001"
    client = MagicMock()
    client.__enter__.return_value = client
    responses = [
        {"folders": [{"id": "online", "online": True}]},
        {"entries": [{"itemId": key, "length": 100, "verifiedBytes": 100}], "nextAfter": None},
        {"size": 100, "mappings": [{"status": "valid", "linkPath": "TV-HD/show/episode.mkv"}]},
    ]
    client.get.side_effect = [MagicMock(status_code=200, json=lambda value=value: value) for value in responses]
    monkeypatch.setattr(infini_cache.requests, "Session", lambda: client)
    paths = infini_cache.fully_cached_paths()
    assert "/library/TV-HD/show/episode.mkv" in paths


def test_sharded_infini_library_link_matches_without_fuse_stat(tmp_path, monkeypatch):
    key = "402c443b-2ffc-4d04-bc81-51d456fa0f5e"
    link = tmp_path / "movie.mkv"
    link.symlink_to("/mnt/remote/infinidysk/.ids/4/0/2/c/4/" + key)
    cached = {"/mnt/remote/infinidysk/.ids/" + key}
    monkeypatch.setattr(infini_cache.os.path, "realpath", lambda path: pytest.fail("Unnecessary FUSE stat"))
    assert infini_cache.item_is_fully_cached(movie(1, file_path=str(link)), cached)
    assert not infini_cache.item_is_fully_cached(movie(1, file_path=str(link)), {"/other/cached"})


def test_active_show_followups_cross_seasons_deduplicate_versions_and_skip_ready():
    from plex_generate_previews.web.priority import EpisodeRow, active_next_episodes
    rows = [EpisodeRow(1, 10, 2, 15), EpisodeRow(2, 10, 2, 16),
            EpisodeRow(3, 10, 2, 17), EpisodeRow(4, 10, 2, 17),
            EpisodeRow(5, 10, 3, 1), EpisodeRow(6, 10, 3, 2),
            EpisodeRow(7, 11, 2, 16)]
    assert active_next_episodes({1}, rows, {3, 4, 5, 6, 7}) == {3: 1, 4: 1, 5: 2, 6: 3}


@pytest.mark.parametrize("full_backlog", [False, True])
def test_completed_current_episode_promotes_followups_above_cached_and_demotes_on_stop(db, monkeypatch, full_backlog):
    from plex_generate_previews.web.priority import EpisodeRow
    scheduler = Scheduler()
    with Session(db) as session:
        session.add(AppSettings(id=1, full_backlog=full_backlog))
        session.add(movie(1, status=PreviewStatus.COMPLETED))
        session.add(movie(2))
        session.add(movie(3, is_priority=True, priority_score=1999, status=PreviewStatus.QUEUED))
        session.add(movie(4, is_priority=True, priority_score=1999,
                          priority_reasons='[{"type":"fully_cached","score":1000},{"type":"on_deck","score":900}]'))
        session.commit()
    monkeypatch.setattr(scheduler, "get_active_playback", lambda: {"available": True, "sessions": [{"rating_key": 1}]})
    monkeypatch.setattr(scheduler, "_read_priority_episode_rows", lambda: [EpisodeRow(1, 10, 2, 15), EpisodeRow(2, 10, 2, 16), EpisodeRow(4, 10, 2, 17)])
    scheduler.refresh_active_playback_metadata()
    with Session(db) as session:
        assert [item.id for item in session.exec(scheduler._queue_candidates(session, 10)).all()] == [2, 4, 3]
        assert session.get(MediaItem, 1).priority_score == 0
        result = asyncio.run(main.get_items(session=session, user="test"))
        assert result['items'][0]['id'] == 2
    monkeypatch.setattr(scheduler, "get_active_playback", lambda: {"available": True, "sessions": []})
    scheduler.refresh_active_playback_metadata()
    with Session(db) as session:
        assert session.get(MediaItem, 2).priority_score == 0
        assert session.get(MediaItem, 4).priority_score == 1900


def test_cached_only_scope_excludes_cold_version_and_full_backlog_restores_it():
    scheduler = Scheduler()
    scheduler.config = SimpleNamespace()
    item = movie(1, priority_reasons='[{"type":"fully_cached","score":1000,"file_paths":["/4k"]}]')
    scheduler._configure_part_scope(item, False)
    assert scheduler.config.preview_part_scopes == {"1": {"/4k"}}
    scheduler._configure_part_scope(item, True)
    assert scheduler.config.preview_part_scopes == {}
    item.priority_reasons='[{"type":"fully_cached","file_paths":["/4k"]},{"type":"active_next"}]'
    scheduler._configure_part_scope(item, False)
    assert scheduler.config.preview_part_scopes == {}


def test_cached_version_with_existing_preview_is_no_longer_pending(tmp_path):
    from plex_generate_previews.web.infini_cache import cached_part_paths
    bif = tmp_path / "preview.bif"; bif.write_bytes(b"ready")
    item = movie(1, media_parts_info=json.dumps([{"file_path":"/4k","bif_path":str(bif)},{"file_path":"/cold"}]))
    assert cached_part_paths(item, {"/4k"}) == {"/4k"}
    assert cached_part_paths(item, {"/4k"}, missing_only=True) == set()


def test_partial_cached_completion_preserves_cold_version_for_full_backlog(db, monkeypatch, tmp_path):
    ready = tmp_path / "ready.bif"; ready.write_bytes(b"ready")
    cold = tmp_path / "cold.bif"
    monkeypatch.setattr(scheduler_module.scheduler, "config", SimpleNamespace(preview_part_scopes={"1": {"/4k"}}, plex_config_folder=str(tmp_path)))
    item = movie(1, status=PreviewStatus.PROCESSING, is_priority=True, priority_score=1000,
                 media_parts_info=json.dumps([{"file_path":"/4k","bif_path":str(ready)},{"file_path":"/cold","bif_path":str(cold)}]))
    scheduler_module.DbProgressManager()._apply_update_to_item(item, {"progress_percent":100})
    assert item.status == PreviewStatus.MISSING
    assert item.is_priority is False
    assert item.priority_score == 0


def test_processing_cached_version_never_opens_cold_version(monkeypatch, tmp_path):
    from plex_generate_previews import media_processing, plex_client
    config = SimpleNamespace(preview_part_scopes={"1": {"/4k"}}, plex_config_folder=str(tmp_path), plex_path_mappings=[],
                             plex_videos_path_mapping="", regenerate_thumbnails=False)
    monkeypatch.setattr(plex_client, "get_media_parts_from_database", lambda *args: [("/cold", "coldhash"), ("/4k", "cachedhash")])
    seen = []
    def isfile(path):
        seen.append(path)
        return path == "/4k"
    monkeypatch.setattr(media_processing.os.path, "isfile", isfile)
    monkeypatch.setattr(media_processing, "_setup_bundle_paths", lambda *args: (str(tmp_path), "output.bif", str(tmp_path)))
    monkeypatch.setattr(media_processing, "_ensure_directories", lambda *args: True)
    monkeypatch.setattr(media_processing, "_cleanup_temp_directory", lambda *args: None)
    generate = MagicMock()
    monkeypatch.setattr(media_processing, "_generate_and_save_bif", generate)
    media_processing.process_item("1", None, None, config, MagicMock())
    assert "/cold" not in seen
    assert generate.call_args.args[0] == "/4k"


def test_pending_list_shows_cached_version_only_and_full_backlog_shows_both(db):
    with Session(db) as session:
        session.add(AppSettings(id=1, full_backlog=False))
        session.add(movie(1, is_priority=True, priority_score=1000,
                          priority_reasons='[{"type":"fully_cached","score":1000,"file_paths":["/4k"]}]',
                          media_parts_info='[{"file_path":"/cold"},{"file_path":"/4k"}]'))
        session.commit()
        result = asyncio.run(main.get_items(session=session, user="test"))
        assert [item["file_path"] for item in result["items"]] == ["/4k"]
        settings = session.get(AppSettings, 1); settings.full_backlog=True;session.add(settings);session.commit()
        result = asyncio.run(main.get_items(session=session, user="test"))
        assert [item["file_path"] for item in result["items"]] == ["/cold", "/4k"]
        assert result["items"][0]["is_priority"] is False
