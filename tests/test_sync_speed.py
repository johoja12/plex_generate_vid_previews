"""Regression checks for bounded media checks and cache catalogue requests."""
import threading
from unittest.mock import MagicMock

import pytest
import requests

from plex_generate_previews.web.scheduler import Scheduler
from plex_generate_previews.web import scheduler as scheduler_module
from plex_generate_previews.web import infini_cache


def test_media_cache_reuses_target_check_but_detects_retarget(tmp_path, monkeypatch):
    first = tmp_path / 'first.mkv'
    first.write_bytes(b'video')
    link = tmp_path / 'episode.mkv'
    link.symlink_to(first)
    scheduler = Scheduler()
    real_exists = scheduler_module.os.path.exists
    calls = []
    def exists(path):
        calls.append(path)
        return real_exists(path)
    monkeypatch.setattr(scheduler_module.os.path, 'exists', exists)
    assert scheduler._sync_media_path_exists(str(link))
    assert scheduler._sync_media_path_exists(str(link))
    assert len(calls) == 1
    link.unlink()
    link.symlink_to(tmp_path / 'missing.mkv')
    assert not scheduler._sync_media_path_exists(str(link))
    assert len(calls) == 2


def test_media_cache_expiry_fresh_and_missing_recovery(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(scheduler_module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setenv('MEDIA_AVAILABILITY_CACHE_SECONDS', '60')
    target = tmp_path / 'target'
    target.touch()
    link = tmp_path / 'link'
    link.symlink_to(target)
    scheduler = Scheduler()
    assert scheduler._sync_media_path_exists(str(link))
    target.unlink()
    assert scheduler._sync_media_path_exists(str(link))
    assert not scheduler._sync_media_path_exists(str(link), fresh=True)
    target.touch()
    assert scheduler._sync_media_path_exists(str(link))
    target.unlink()
    clock[0] += 61
    assert not scheduler._sync_media_path_exists(str(link))
    target.touch()
    assert scheduler._sync_media_path_exists(str(link))
    link.unlink()
    assert not scheduler._sync_media_path_exists(str(link))


@pytest.mark.parametrize('fail', [False, True])
def test_parallel_catalogue_is_bounded_complete_and_fails_closed(monkeypatch, fail):
    monkeypatch.setenv('INFINIDYSK_URL', 'http://infini')
    monkeypatch.setenv('INFINIDYSK_API_KEY', 'key')
    keys = [f'00000000-0000-0000-0000-{i:012d}' for i in range(16)]
    lock = threading.Lock()
    clients = []
    requested = []
    def response(data):
        return MagicMock(status_code=200, json=lambda: data)
    class Client:
        def __init__(self):
            self.headers = {}
            self.owner = None
            clients.append(self)
        def __enter__(self):
            self.owner = threading.get_ident()
            return self
        def __exit__(self, *args):
            pass
        def get(self, url, params, timeout):
            assert threading.get_ident() == self.owner
            if url.endswith('/api/native-cache'):
                return response({'folders': [{'id': 'online', 'online': True}]})
            if url.endswith('/api/native-cache/entries'):
                return response({'entries': [{'itemId': key, 'length': 10, 'verifiedBytes': 10} for key in keys]})
            key = params['davItemId']
            with lock:
                requested.append(key)
            if fail and key == keys[0]:
                raise requests.ConnectionError('private')
            return response({'size': 10, 'mappings': [{'status': 'valid', 'linkPath': '/mnt/plex/' + key}]})
    monkeypatch.setattr(infini_cache.requests, 'Session', Client)
    paths = infini_cache.fully_cached_paths()
    assert len(clients) <= 9  # one catalogue session plus eight detail sessions
    if fail:
        assert paths == set()
    else:
        assert set(requested) == set(keys)
        assert paths == {prefix + key for key in keys for prefix in ('/mnt/remote/infinidysk/.ids/', '/mnt/plex/')}


@pytest.mark.parametrize('case', ['new', 'changed_hash', 'changed_path', 'missing', 'unchanged'])
def test_sync_forces_new_changed_and_missing_media_checks(mock_config, monkeypatch, case):
    import json
    from datetime import datetime
    from plex_generate_previews.web.models import MediaItem, MediaType, PreviewStatus
    scheduler = Scheduler()
    scheduler.config = mock_config
    monkeypatch.setattr(scheduler, '_get_bif_path', lambda value: '/bif/' + value)
    calls = []
    monkeypatch.setattr(scheduler, '_sync_media_path_exists',
                        lambda path, fresh=False: calls.append(fresh) or True)
    parts = json.dumps([{'file_path': '/episode', 'bundle_hash': 'hash', 'bif_path': '/bif/hash'}])
    row = MediaItem(id=1, title='Episode', media_type=MediaType.EPISODE,
                    library_name='TV', bundle_hash='hash', media_parts_info=parts,
                    status=PreviewStatus.MISSING)
    if case == 'new':
        row = None
    elif case == 'changed_hash':
        row.bundle_hash = 'oldhash'
    elif case == 'changed_path':
        row.media_parts_info = parts.replace('/episode', '/old')
    elif case == 'missing':
        row.status = PreviewStatus.MEDIA_MISSING
    scheduler._process_sync_item('TV', 1, 'Episode', 'episode', 'hash', datetime(2026, 1, 1),
                                 [('/episode', 'hash')], {}, {}, row)
    assert calls == [case != 'unchanged']
