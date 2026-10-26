import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock
import pytest
from plex_generate_previews import media_processing as mp
from plex_generate_previews.web.models import MediaItem, MediaType, PreviewStatus
from plex_generate_previews.web.scheduler import Scheduler

@pytest.mark.parametrize('running_poll', [False, True])
def test_invalid_utf8_diagnostic_does_not_abort_generation(tmp_path, mock_config, monkeypatch, running_poll):
    monkeypatch.setattr(mp, 'detect_hdr_ffprobe', lambda *a, **kw: False)
    monkeypatch.setattr(mp, 'heuristic_allows_skip', lambda *a, **kw: False)
    monkeypatch.setattr(mp.time, 'sleep', lambda *a: None)
    folder = tmp_path / 'images'
    class Process:
        returncode = 0
        def __init__(self): self.calls = 0
        def poll(self):
            self.calls += 1
            return None if running_poll and self.calls == 1 else 0
    def popen(args, stderr, stdout):
        stderr.buffer.write(b'container diagnostic: \xc5\n')
        stderr.flush()
        (folder / 'img-000001.jpg').write_bytes(b'\xff\xd8\xff\xd9')
        return Process()
    monkeypatch.setattr(mp.subprocess, 'Popen', popen)
    result = mp.generate_images('/generic/media.mkv', str(folder), None, None, mock_config)
    assert result[0] is True
    assert result[1] == 1
    assert (folder / '0000000000.jpg').exists()

@pytest.mark.parametrize('added_changed', [False, True])
@pytest.mark.parametrize('media_exists', [False, True])
def test_media_availability_is_independent_of_added_timestamp(mock_config, monkeypatch, added_changed, media_exists):
    scheduler = Scheduler()
    scheduler.config = mock_config
    stamp = datetime(2026, 1, 1)
    item = MediaItem(id=80001, title='Generic episode', media_type=MediaType.EPISODE,
                     library_name='Generic library', status=PreviewStatus.MEDIA_MISSING if media_exists else PreviewStatus.FAILED,
                     error_message='Previous failure', added_at=stamp, progress=0)
    monkeypatch.setattr(os.path, 'exists', lambda path: media_exists)
    updated, _ = scheduler._process_sync_item('Generic library', item.id, item.title, 'episode',
        'a' * 40, stamp + timedelta(days=1) if added_changed else stamp,
        [('/generic/media.mkv', 'a' * 40)], {}, {}, item)
    assert updated.status == (PreviewStatus.MISSING if media_exists else PreviewStatus.MEDIA_MISSING)
    if media_exists:
        assert updated.error_message is None
