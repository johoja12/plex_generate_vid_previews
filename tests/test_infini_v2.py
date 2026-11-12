"""Contract tests against the deployed Rust v2 shapes (14d8d1a)."""
import copy
from unittest.mock import MagicMock

import pytest
import requests

from plex_generate_previews.web import infini_cache

ID = '00000000-0000-0000-0000-000000000001'


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv('INFINIDYSK_URL', 'http://infini')
    monkeypatch.setenv('INFINIDYSK_API_KEY', 'test-key')
    monkeypatch.setenv('INFINIDYSK_API_VERSION', '2')
    responses = {
        '/api/v1/native-cache': {'configured': True, 'available': True, 'recovery_mode': False, 'reason': None},
        '/api/v1/native-cache/diagnostics': [],
        '/api/v1/native-cache/legacy/diagnostics': [{'id': 'disk', 'identity_ok': True}],
        '/api/v1/native-cache/files': [],
        '/api/v1/native-cache/legacy': {'folders': [{'id': 'disk'}], 'entries': [
            {'item': ID.replace('-', ''), 'source_present': True, 'folder': 'disk', 'size': 100,
             'cached_bytes': 100, 'pending_bytes': 0, 'dirty': 1}], 'next': None},
        '/api/v1/files/' + ID: {
            'entry': {'item_id': ID, 'kind': 'file', 'state': 'healthy', 'size': 100, 'path': '/content/video.mkv'},
            'source_revision': 'current', 'cache_available': True, 'cache_warning': None,
            'cached_ranges': [{'start': 0, 'end': 60}, {'start': 60, 'end': 100}],
            'links': ['/mnt/remote/infinidysk/.ids/0/0/0/0/0/' + ID],
            'library_links': [{'item': ID, 'stale': False, 'link_kind': 'symlink', 'path': '/mnt/plex/video.mkv'}]},
    }
    calls = []
    class Client:
        def __init__(self):
            self.headers = {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, url, params, timeout):
            assert self.headers['x-api-key'] == 'test-key'
            path = url.removeprefix('http://infini')
            calls.append((path, params))
            value = responses[path]
            if isinstance(value, Exception):
                raise value
            if callable(value):
                value = value(params)
            if isinstance(value, int):
                result = MagicMock(status_code=value)
                result.raise_for_status.side_effect = requests.HTTPError('private')
                return result
            return MagicMock(status_code=200, json=lambda: copy.deepcopy(value))
    monkeypatch.setattr(infini_cache.requests, 'Session', Client)
    return responses, calls


def test_v2_current_coverage_and_valid_links(api):
    assert infini_cache.fully_cached_paths() == {
        '/mnt/remote/infinidysk/.ids/' + ID,
        '/mnt/remote/infinidysk/.ids/0/0/0/0/0/' + ID,
        '/mnt/remote/infinidysk/content/video.mkv', '/mnt/plex/video.mkv'}


@pytest.mark.parametrize('mutation', [
    'gap', 'warning', 'missing_revision', 'wrong_id', 'wrong_size', 'offline',
    'unknown_storage', 'unavailable', 'recovery', 'pending', 'missing_source',
    'malformed_range', 'bool_size', 'request_failure', 'auth_failure', 'bad_page',
])
def test_v2_incomplete_unknown_and_unavailable_fail_closed(api, mutation):
    data, _ = api
    detail = data['/api/v1/files/' + ID]
    entry = data['/api/v1/native-cache/legacy']['entries'][0]
    if mutation == 'gap': detail['cached_ranges'][1]['start'] = 61
    elif mutation == 'warning': detail['cache_warning'] = 'Coverage could not be verified'
    elif mutation == 'missing_revision': detail['source_revision'] = None
    elif mutation == 'wrong_id': detail['entry']['item_id'] = ID[:-1] + '2'
    elif mutation == 'wrong_size': detail['entry']['size'] = 101
    elif mutation == 'offline': data['/api/v1/native-cache/legacy/diagnostics'][0]['identity_ok'] = False
    elif mutation == 'unknown_storage': data['/api/v1/native-cache/legacy/diagnostics'][0]['identity_ok'] = None
    elif mutation == 'unavailable': data['/api/v1/native-cache']['available'] = False
    elif mutation == 'recovery': data['/api/v1/native-cache']['recovery_mode'] = True
    elif mutation == 'pending': entry['pending_bytes'] = 1
    elif mutation == 'missing_source': entry['source_present'] = False
    elif mutation == 'malformed_range': detail['cached_ranges'] = [{'start': -1, 'end': 100}]
    elif mutation == 'bool_size': detail['entry']['size'] = True
    elif mutation == 'request_failure': data['/api/v1/files/' + ID] = requests.ConnectionError('private')
    elif mutation == 'auth_failure': data['/api/v1/native-cache'] = 401
    elif mutation == 'bad_page': data['/api/v1/native-cache/legacy']['entries'] = None
    assert infini_cache.fully_cached_paths() == set()


def test_v2_stale_library_link_excluded_but_identity_can_match_local_symlink(api):
    data, _ = api
    data['/api/v1/files/' + ID]['library_links'][0]['stale'] = True
    paths = infini_cache.fully_cached_paths()
    assert '/mnt/plex/video.mkv' not in paths
    assert '/mnt/remote/infinidysk/.ids/' + ID in paths


def test_v2_original_cache_paginates_and_rejects_repeated_cursor(api):
    data, calls = api
    page = copy.deepcopy(data['/api/v1/native-cache/legacy'])
    data['/api/v1/native-cache/legacy'] = lambda params: dict(page, next='cursor') if not params['after'] else dict(page, next=None)
    assert infini_cache.fully_cached_paths()
    assert [params['after'] for path, params in calls if path.endswith('/legacy')] == ['', 'cursor']
    data['/api/v1/native-cache/legacy'] = dict(page, next='cursor')
    assert infini_cache.fully_cached_paths() == set()


@pytest.mark.parametrize('revision,expected', [('current', True), ('old', False)])
def test_v2_native_store_requires_current_revision_and_handles_absent_original(api, revision, expected):
    data, _ = api
    data['/api/v1/native-cache/legacy'] = 404
    data['/api/v1/native-cache/legacy/diagnostics'] = 404
    data['/api/v1/native-cache/diagnostics'] = [{'folder_id': 'new', 'identity_ok': True}]
    data['/api/v1/native-cache/files'] = [{'item': ID, 'revision': revision, 'size_bytes': 100, 'cached_bytes': 100, 'folder_id': 'new'}]
    assert bool(infini_cache.fully_cached_paths()) is expected


def test_v2_limits_details_to_scheduler_candidates(api):
    from plex_generate_previews.web.infini_v2 import fully_cached_paths_v2
    _, calls = api
    assert fully_cached_paths_v2('http://infini', 'test-key', {ID[:-1] + '2'}) == set()
    assert not any('/api/v1/files/' in path for path, _ in calls)


def test_v2_candidate_ids_read_links_without_following_remote_mounts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from plex_generate_previews.web.infini_v2 import candidate_ids
    import json
    link = tmp_path / 'video.mkv'
    link.symlink_to('/mnt/remote/infinidysk/.ids/0/0/0/0/0/' + ID)
    other = tmp_path / 'other.mkv'
    other.symlink_to('/mnt/remote/debrid/video.mkv')
    item = SimpleNamespace(media_parts_info=json.dumps([{'file_path': str(link)}, {'file_path': str(other)}]))
    assert candidate_ids([item]) == {ID}
    # A pointer form that cannot be narrowed retains the complete lookup.
    link.unlink()
    link.symlink_to('/mnt/remote/infinidysk/content/video.mkv')
    assert candidate_ids([item]) is None
