import os
import pytest
from unittest.mock import MagicMock
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
