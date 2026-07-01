import os
import pytest
from unittest.mock import MagicMock
from sqlalchemy import inspect
from sqlmodel import Session, SQLModel, create_engine, select
from sqlmodel.pool import StaticPool
from plex_generate_previews.web.models import MediaItem, MediaType, PreviewStatus
from plex_generate_previews.web.priority import PriorityInfo, add_reason
from plex_generate_previews.web.main import priority_payload
from plex_generate_previews.web.scheduler import Scheduler
from plex_generate_previews.config import Config


def test_scan_filesystem_for_bifs_with_stripped_hash_bif(temp_dir, mock_config):
    """
    Test that _scan_filesystem_for_bifs correctly identifies a BIF file
    using Plex's stripped hash naming convention, after the fix.
    """
    # Simulate the Plex config folder structure
    plex_config_path = os.path.join(temp_dir, "plex_config")
    mock_config.plex_config_folder = plex_config_path

    # Create the mock BIF file path
    # Full hash: 98ee3ecdf4aba34b7708a7d8c92422f4e96eebf2 (from the user's example)
    # This means the directory is '9' and the filename is '8ee3ecdf4aba34b7708a7d8c92422f4e96eebf2.bundle'
    bundle_dir = "9"
    bundle_filename_stem = "8ee3ecdf4aba34b7708a7d8c92422f4e96eebf2" # This is the 39-char stripped hash
    full_bundle_hash = bundle_dir + bundle_filename_stem # This is the 40-char full hash

    mock_bif_path = os.path.join(
        plex_config_path,
        "Media",
        "localhost",
        bundle_dir,
        f"{bundle_filename_stem}.bundle",
        "Contents",
        "Indexes",
        "index-sd.bif"
    )

    # Ensure the directory structure exists and create a dummy BIF file
    os.makedirs(os.path.dirname(mock_bif_path), exist_ok=True)
    with open(mock_bif_path, "w") as f:
        f.write("dummy bif content")

    # Instantiate the Scheduler without arguments and set config
    scheduler = Scheduler()
    scheduler.config = mock_config

    # Call the method under test
    bundle_hash_map = scheduler._scan_filesystem_for_bifs()

    # Assert that the full_bundle_hash is found in the map and points to the correct path
    assert full_bundle_hash in bundle_hash_map
    assert bundle_hash_map[full_bundle_hash] == mock_bif_path

    # Test with a file that previously would have failed due to "e" vs "7" mismatch
    # Simulate: full hash e730b9ff... in directory "e"
    bug_bundle_dir = "e"
    bug_bundle_filename_stem = "730b9ff1874a14b1e4f09f671a1f278531d04fb"
    bug_full_bundle_hash = bug_bundle_dir + bug_bundle_filename_stem

    bug_mock_bif_path = os.path.join(
        plex_config_path,
        "Media",
        "localhost",
        bug_bundle_dir,
        f"{bug_bundle_filename_stem}.bundle",
        "Contents",
        "Indexes",
        "index-sd.bif"
    )

    os.makedirs(os.path.dirname(bug_mock_bif_path), exist_ok=True)
    with open(bug_mock_bif_path, "w") as f:
        f.write("dummy bug bif content")

    scheduler_retest = Scheduler()
    scheduler_retest.config = mock_config
    bundle_hash_map_retest = scheduler_retest._scan_filesystem_for_bifs()

    assert bug_full_bundle_hash in bundle_hash_map_retest
    assert bundle_hash_map_retest[bug_full_bundle_hash] == bug_mock_bif_path


def test_mediaitem_has_priority_score_metadata_columns():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    columns = {column["name"] for column in inspect(engine).get_columns("mediaitem")}

    assert "priority_score" in columns
    assert "priority_reasons" in columns
    assert "priority_last_calculated_at" in columns


def test_apply_priority_info_sets_score_reasons_and_compatibility_flag():
    scheduler = Scheduler()
    item = MediaItem(
        id=1,
        title="Episode",
        library_name="TV",
        media_type=MediaType.EPISODE,
    )
    info = PriorityInfo()
    add_reason(info, "next_episode", 800, account_id=10)

    changed = scheduler._apply_priority_info_to_item(item, info)

    assert changed is True
    assert item.is_priority is True
    assert item.priority_score == 800
    assert item.priority_reasons == '[{"type":"next_episode","score":800,"account_id":10}]'
    assert item.priority_last_calculated_at is not None


def test_apply_priority_info_clears_missing_priority():
    scheduler = Scheduler()
    item = MediaItem(
        id=1,
        title="Episode",
        library_name="TV",
        media_type=MediaType.EPISODE,
        is_priority=True,
        priority_score=800,
        priority_reasons="[]",
    )

    changed = scheduler._apply_priority_info_to_item(item, None)

    assert changed is True
    assert item.is_priority is False
    assert item.priority_score == 0
    assert item.priority_reasons is None
    assert item.priority_last_calculated_at is None


def test_priority_ordering_uses_score_before_updated_at():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        low_score_newer = MediaItem(
            id=1,
            title="Low",
            library_name="TV",
            media_type=MediaType.EPISODE,
            status=PreviewStatus.MISSING,
            priority_score=100,
        )
        high_score_older = MediaItem(
            id=2,
            title="High",
            library_name="TV",
            media_type=MediaType.EPISODE,
            status=PreviewStatus.MISSING,
            priority_score=800,
        )
        session.add(low_score_newer)
        session.add(high_score_older)
        session.commit()

        rows = session.exec(
            select(MediaItem)
            .where(MediaItem.status == PreviewStatus.MISSING)
            .order_by(*Scheduler._priority_order_by())
        ).all()

    assert [row.id for row in rows] == [2, 1]


def test_clear_completed_priority_metadata_clears_all_priority_fields():
    scheduler = Scheduler()
    item = MediaItem(
        id=1,
        title="Done",
        library_name="TV",
        media_type=MediaType.EPISODE,
        status=PreviewStatus.COMPLETED,
        is_priority=True,
        priority_score=800,
        priority_reasons="[]",
    )

    changed = scheduler._clear_completed_priority_metadata(item)

    assert changed is True
    assert item.is_priority is False
    assert item.priority_score == 0
    assert item.priority_reasons is None
    assert item.priority_last_calculated_at is None


def test_priority_payload_exposes_score_and_reasons():
    item = MediaItem(
        id=1,
        title="Episode",
        library_name="TV",
        media_type=MediaType.EPISODE,
        is_priority=True,
        priority_score=800,
        priority_reasons='[{"type":"next_episode","score":800}]',
    )

    assert priority_payload(item) == {
        "is_priority": True,
        "priority_score": 800,
        "priority_reasons": [{"type": "next_episode", "score": 800}],
    }


def test_priority_payload_handles_missing_or_invalid_reasons():
    missing = MediaItem(
        id=1,
        title="Missing",
        library_name="TV",
        media_type=MediaType.EPISODE,
    )
    invalid = MediaItem(
        id=2,
        title="Invalid",
        library_name="TV",
        media_type=MediaType.EPISODE,
        priority_reasons="not-json",
    )

    assert priority_payload(missing)["priority_reasons"] == []
    assert priority_payload(invalid)["priority_reasons"] == []
