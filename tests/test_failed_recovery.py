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
    monkeypatch.setattr(mp, 'probe_video_duration', lambda *a, **kw: None)
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


@pytest.mark.parametrize('streams, expected', [
    ([{'duration': '1708.209'}], 1708.209),
    ([{'tags': {'DURATION': '00:25:31.334000000'}}], 1531.334),
    ([{'duration': 'N/A', 'tags': {'DURATION': '00:28:28.209000000'}}], 1708.209),
    ([{'duration': 'nan'}], None),
    ([{'duration': '-1'}], None),
    ([{'tags': {'DURATION': '00:99:00'}}], None),
    ([{'duration': '100'}, {'duration': '200'}], None),
    ([], None),
])
def test_expected_duration_uses_only_unambiguous_video_timeline(monkeypatch, streams, expected):
    import json
    def run(args, **kwargs):
        assert args[args.index('-select_streams') + 1] == 'v'
        assert kwargs['timeout'] == 10.0
        return MagicMock(returncode=0, stdout=json.dumps({'streams': streams}))
    monkeypatch.setattr(mp.subprocess, 'run', run)
    assert mp.probe_video_duration('/generic/media.mkv', 'ffprobe') == expected


@pytest.mark.parametrize('error', [mp.subprocess.TimeoutExpired('ffprobe', 10), OSError('unavailable')])
def test_video_duration_probe_failure_retains_container_fallback(monkeypatch, error):
    def run(*args, **kwargs):
        raise error
    monkeypatch.setattr(mp.subprocess, 'run', run)
    assert mp.probe_video_duration('/generic/media.mkv', 'ffprobe') is None


@pytest.mark.parametrize('video_duration, succeeds', [(10.0, True), (100.0, False), (None, False)])
def test_long_subtitle_timeline_does_not_require_extra_video_frames(tmp_path, mock_config, monkeypatch, video_duration, succeeds):
    monkeypatch.setattr(mp, 'detect_hdr_ffprobe', lambda *a, **kw: False)
    monkeypatch.setattr(mp, 'probe_video_duration', lambda *a, **kw: video_duration)
    monkeypatch.setattr(mp, 'heuristic_allows_skip', lambda *a, **kw: False)
    folder = tmp_path / 'images'
    mock_config.plex_bif_frame_interval = 5
    def popen(args, stderr, stdout):
        stderr.write('Duration: 00:01:40.00\n')
        stderr.flush()
        for i in range(2):
            (folder / f'img-{i:06d}.jpg').write_bytes(b'\xff\xd8\xff\xd9')
        return MagicMock(returncode=0, poll=lambda: 0)
    monkeypatch.setattr(mp.subprocess, 'Popen', popen)
    assert mp.generate_images('/generic/media.mkv', str(folder), None, None, mock_config)[0] is succeeds
