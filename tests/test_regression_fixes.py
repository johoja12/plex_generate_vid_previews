"""Regression tests for production bugs fixed in October 2026.

Each test guards a failure mode that broke the live deployment. These cover
the reliability-critical paths (sync, path handling, mount validation) that
the rest of the suite does not exercise.
"""

import os
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from plex_generate_previews.web import scheduler as scheduler_module
from plex_generate_previews.web.models import AppSettings, MediaItem, MediaType, PreviewStatus
from plex_generate_previews.web.scheduler import Scheduler, _ensure_aware


def _in_memory_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return engine


# ---------------------------------------------------------------------------
# Naive datetime handling.
#
# Regression: naive datetimes from Plex / older DB rows crashed the sync loop
# (TypeError on naive/aware comparison) and were rejected at DB flush time
# (ValueError: "Datetime values must have timezone information").
# ---------------------------------------------------------------------------

def test_ensure_aware_converts_naive_to_utc():
    naive = datetime(2026, 10, 14, 12, 30, 0)
    result = _ensure_aware(naive)
    assert result.tzinfo == timezone.utc
    assert result.replace(tzinfo=None) == naive


def test_ensure_aware_preserves_aware_datetime():
    aware = datetime(2026, 10, 14, 12, 30, 0, tzinfo=timezone.utc)
    assert _ensure_aware(aware) == aware


def test_ensure_aware_passes_through_none():
    assert _ensure_aware(None) is None


def test_process_sync_item_normalizes_naive_added_at(mock_config):
    """A naive added_at (as Plex hands back) must not crash the sync item
    path and must produce a DB-writable timezone-aware timestamp."""
    scheduler = Scheduler()
    scheduler.config = mock_config

    naive_added_at = datetime(2026, 10, 14, 12, 0, 0)  # naive, like Plex returns
    item, stat_type = scheduler._process_sync_item(
        section_title="Movies",
        item_key=101,
        title="Test Movie",
        item_type="movie",
        bundle_hash=None,
        added_at=naive_added_at,
        media_parts=[],
        bundle_hash_map={},
        priority_item_keys={},
        db_item=None,
    )

    assert item.added_at.tzinfo is not None
    assert item.added_at == naive_added_at.replace(tzinfo=timezone.utc)
    assert stat_type == "movies_added"

    # The exact production failure: flushing a naive datetime raised
    # ValueError at the DB bind layer. This must succeed.
    engine = _in_memory_engine()
    with Session(engine) as session:
        session.add(item)
        session.commit()


def test_process_sync_item_update_handles_naive_db_added_at(mock_config):
    """An existing row whose added_at came back naive must not break the
    update comparison against the aware timestamp from Plex."""
    scheduler = Scheduler()
    scheduler.config = mock_config

    db_item = MediaItem(
        id=102,
        title="Test Movie",
        media_type=MediaType.MOVIE,
        library_name="Movies",
        status=PreviewStatus.MISSING,
        added_at=datetime(2026, 10, 13, 12, 0, 0, tzinfo=timezone.utc),
    )
    # Simulate an older row read back as naive (as SQLite/Plex once produced)
    db_item.added_at = datetime(2026, 10, 13, 12, 0, 0)

    plex_added_at = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
    updated, stat_type = scheduler._process_sync_item(
        section_title="Movies",
        item_key=102,
        title="Test Movie",
        item_type="movie",
        bundle_hash=None,
        added_at=plex_added_at,
        media_parts=[],
        bundle_hash_map={},
        priority_item_keys={},
        db_item=db_item,
    )

    assert updated.added_at == plex_added_at
    assert updated.added_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Plex path translation before existence checks.
#
# Regression: media existence was checked against the raw Plex-side path
# (e.g. /data/...) which does not exist inside the container, so every item
# was flagged MEDIA_MISSING. Paths must be translated via the configured
# mappings (e.g. /data -> /media) before checking.
# ---------------------------------------------------------------------------

def test_translate_plex_path_prefers_longest_mapping_prefix(mock_config):
    scheduler = Scheduler()
    scheduler.config = mock_config
    mock_config.plex_path_mappings = [
        ("/data", "/media"),
        ("/data/movies", "/mnt/movies"),
    ]

    assert scheduler._translate_plex_path("/data/movies/Film.mkv") == "/mnt/movies/Film.mkv"
    assert scheduler._translate_plex_path("/data/tv/Show.mkv") == "/media/tv/Show.mkv"


def test_translate_plex_path_falls_back_to_legacy_single_mapping(mock_config):
    scheduler = Scheduler()
    scheduler.config = mock_config
    mock_config.plex_path_mappings = []
    mock_config.plex_videos_path_mapping = "/data"
    mock_config.plex_local_videos_path_mapping = "/media"

    assert scheduler._translate_plex_path("/data/Film.mkv") == "/media/Film.mkv"


def test_translate_plex_path_leaves_unmapped_paths_unchanged(mock_config):
    scheduler = Scheduler()
    scheduler.config = mock_config
    mock_config.plex_path_mappings = [("/data", "/media")]

    assert scheduler._translate_plex_path("/other/Film.mkv") == "/other/Film.mkv"
    assert scheduler._translate_plex_path("") == ""


def test_process_sync_item_checks_translated_path_not_plex_path(mock_config, temp_dir):
    """The 100% MEDIA_MISSING regression: with a /data -> <local> mapping and
    the file present locally, the item must NOT be flagged MEDIA_MISSING even
    though the raw Plex path (/data/...) does not exist."""
    scheduler = Scheduler()
    scheduler.config = mock_config
    mock_config.plex_path_mappings = [("/data", temp_dir)]

    real_file = os.path.join(temp_dir, "movies", "Film.mkv")
    os.makedirs(os.path.dirname(real_file), exist_ok=True)
    with open(real_file, "w") as f:
        f.write("fake media")

    item, _ = scheduler._process_sync_item(
        section_title="Movies",
        item_key=103,
        title="Film",
        item_type="movie",
        bundle_hash=None,
        added_at=datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc),
        media_parts=[("/data/movies/Film.mkv", None)],
        bundle_hash_map={},
        priority_item_keys={},
        db_item=None,
    )
    assert item.status != PreviewStatus.MEDIA_MISSING

    # Control: without the mapping, the untranslated Plex path does not exist
    # locally, so MEDIA_MISSING is the correct verdict.
    mock_config.plex_path_mappings = []
    missing_item, _ = scheduler._process_sync_item(
        section_title="Movies",
        item_key=104,
        title="Film",
        item_type="movie",
        bundle_hash=None,
        added_at=datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc),
        media_parts=[("/data/movies/Film.mkv", None)],
        bundle_hash_map={},
        priority_item_keys={},
        db_item=None,
    )
    assert missing_item.status == PreviewStatus.MEDIA_MISSING


# ---------------------------------------------------------------------------
# Mount validation.
#
# Regression: the mount check hardcoded the upstream author's personal paths
# (/mnt/plex, ...), which blocked sync for everyone else. Only explicitly
# configured paths plus the Plex config folder may be checked.
# ---------------------------------------------------------------------------

def test_build_mount_check_paths_contains_no_hardcoded_upstream_paths(mock_config):
    scheduler = Scheduler()
    scheduler.config = mock_config
    mock_config.plex_config_folder = "/plex-config"

    paths = scheduler._build_mount_check_paths("/mnt/user/data,/mnt/user/media")

    assert paths == ["/mnt/user/data", "/mnt/user/media", "/plex-config"]
    assert "/mnt/plex" not in paths


def test_build_mount_check_paths_with_no_configured_paths(mock_config):
    scheduler = Scheduler()
    scheduler.config = mock_config
    mock_config.plex_config_folder = "/plex-config"

    assert scheduler._build_mount_check_paths("") == ["/plex-config"]
    assert scheduler._build_mount_check_paths(None) == ["/plex-config"]


def _scheduler_with_settings(monkeypatch, mock_config, tmp_path=None, enabled=True, paths=None):
    engine = _in_memory_engine()
    with Session(engine) as session:
        session.add(AppSettings(
            id=1,
            mount_check_enabled=enabled,
            mount_check_paths=paths,
        ))
        session.commit()
    monkeypatch.setattr(scheduler_module, "engine", engine)
    scheduler = Scheduler()
    scheduler.config = mock_config
    mock_config.plex_config_folder = str(tmp_path) if tmp_path else "/plex-config"
    return scheduler


def test_validate_mount_paths_flags_missing_configured_path(monkeypatch, mock_config, temp_dir):
    scheduler = _scheduler_with_settings(
        monkeypatch, mock_config, temp_dir,
        paths="/nonexistent/plex-mount-xyz",
    )
    valid, missing = scheduler.validate_mount_paths()
    assert valid is False
    assert "/nonexistent/plex-mount-xyz" in missing


def test_validate_mount_paths_passes_for_existing_paths(monkeypatch, mock_config, temp_dir):
    scheduler = _scheduler_with_settings(
        monkeypatch, mock_config, temp_dir,
        paths=temp_dir,
    )
    valid, missing = scheduler.validate_mount_paths()
    assert valid is True
    assert missing == []


def test_validate_mount_paths_disabled_skips_checks(monkeypatch, mock_config, temp_dir):
    scheduler = _scheduler_with_settings(
        monkeypatch, mock_config, temp_dir,
        enabled=False,
        paths="/nonexistent/plex-mount-xyz",
    )
    valid, missing = scheduler.validate_mount_paths()
    assert valid is True
    assert missing == []
