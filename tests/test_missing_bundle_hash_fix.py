
import pytest
from unittest.mock import MagicMock, patch
import xml.etree.ElementTree as ET
from plex_generate_previews.media_processing import process_item
from plex_generate_previews.config import Config

# XML without 'hash' attribute in MediaPart
XML_MISSING_HASH = """<?xml version="1.0" encoding="UTF-8"?>
<MediaContainer size="1">
    <Video ratingKey="54321" key="/library/metadata/54321" type="movie" title="Test Movie">
        <Media id="11111" duration="7200000">
            <MediaPart id="22222" key="/library/parts/22222/1234567890/file.mkv" duration="7200000" file="/data/movies/Test Movie (2024)/Test Movie (2024).mkv">
                <Stream id="1" streamType="1" codec="h264" index="0" />
            </MediaPart>
        </Media>
    </Video>
</MediaContainer>
"""

@patch('plex_generate_previews.plex_client.get_media_parts_from_database')
def test_process_item_raises_when_no_hash(mock_get_media_parts, mock_config):
    """Test that process_item raises RuntimeError when no bundle hash is found."""
    mock_plex = MagicMock()
    mock_plex.query.return_value = ET.fromstring(XML_MISSING_HASH)
    mock_get_media_parts.return_value = [
        ("/data/movies/Test Movie (2024)/Test Movie (2024).mkv", None)
    ]
    
    # Mock file existence so it doesn't fail on "file not found"
    with patch('os.path.isfile', return_value=True):
        # We expect a RuntimeError because no parts were processed
        with pytest.raises(RuntimeError, match="Invalid or missing bundle hash"):
            process_item("54321", None, None, mock_config, mock_plex)
