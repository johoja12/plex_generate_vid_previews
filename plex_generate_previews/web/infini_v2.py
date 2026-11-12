"""Read the Rust v2 admin API; catalogue byte counts are candidates, not proof."""
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import requests


def candidate_ids(items):
    """Resolve only local library pointers, without statting FUSE targets.

    The scheduler needs cache eligibility for its candidates, not the complete
    InfiniDysk library. Unknown pointer forms fall back to the full catalogue.
    """
    mount = os.environ.get('INFINIDYSK_MOUNT_PATH', '/mnt/remote/infinidysk').rstrip('/')
    paths = set()
    for item in items:
        try:
            parts = json.loads(item.media_parts_info) if item.media_parts_info else [{'file_path': item.file_path}]
            paths.update(p['file_path'] for p in parts if isinstance(p.get('file_path'), str))
        except (ValueError, TypeError, AttributeError, KeyError):
            continue

    def resolve_batch(batch):
        ids = set()
        for path in batch:
            if not path.startswith(mount + '/'):
                try:
                    target = os.readlink(path)
                    path = target if os.path.isabs(target) else os.path.normpath(os.path.join(os.path.dirname(path), target))
                except OSError:
                    if not os.path.lexists(path):
                        continue  # Missing library entries cannot be processed.
                    # A regular file may be a registered mapping; retain the
                    # full lookup rather than silently dropping such candidates.
                    return None
            if not path.startswith(mount + '/'):
                continue  # Other providers cannot use InfiniDysk's native cache.
            if not path.startswith(mount + '/.ids/'):
                return None
            match = re.fullmatch(r'([0-9a-fA-F-]{32,36})(?:\.[^/]*)?', os.path.basename(path))
            if not match:
                return None
            try:
                ids.add(str(UUID(match[1])))
            except ValueError:
                return None
        return ids

    paths = list(paths)
    with ThreadPoolExecutor(max_workers=8) as executor:
        batches = list(executor.map(resolve_batch, [paths[i::8] for i in range(8)]))
    return None if any(batch is None for batch in batches) else set().union(*batches)


def _get(client, url, path, optional=False, **params):
    response = client.get(url + path, params=params, timeout=(5, 15))
    if optional and response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def _complete_ranges(ranges, size):
    if not isinstance(ranges, list) or not ranges:
        return False
    intervals = []
    for part in ranges:
        if not isinstance(part, dict):
            return False
        start, end = part.get('start'), part.get('end')
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= size:
            return False
        intervals.append((start, end))
    covered = 0
    for start, end in sorted(intervals):
        if start > covered:
            return False
        covered = max(covered, end)
    return covered == size


def _verified_paths(details, item_id, candidates, mount, library):
    if not isinstance(details, dict):
        return set()
    entry = details.get('entry')
    if not isinstance(entry, dict):
        return set()
    size = entry.get('size')
    revision = details.get('source_revision')
    try:
        identity = str(UUID(entry.get('item_id', '')))
    except (ValueError, TypeError, AttributeError):
        return set()
    if (identity != item_id or entry.get('kind') != 'file'
            or entry.get('state') == 'hidden' or type(size) is not int or size <= 0
            or not isinstance(revision, str) or not revision
            or not any(size == length and (expected is None or revision == expected)
                       for length, expected in candidates)
            or details.get('cache_available') is not True
            or 'cache_warning' not in details or details['cache_warning'] is not None
            or not _complete_ranges(details.get('cached_ranges'), size)):
        return set()
    paths = {mount + '/.ids/' + item_id}
    content = entry.get('path')
    if isinstance(content, str) and content.startswith('/content/') and os.path.normpath(content) == content:
        paths.add(mount + content)
    for link in details.get('links') or []:
        if isinstance(link, str) and link.startswith(mount + '/') and os.path.normpath(link) == link:
            paths.add(link)
    for link in details.get('library_links') or []:
        if not isinstance(link, dict):
            continue
        path = link.get('path')
        if (link.get('stale') is False and link.get('link_kind') == 'symlink'
                and link.get('item') == item_id and isinstance(path, str)
                and path.startswith(library.rstrip('/') + '/') and os.path.normpath(path) == path):
            paths.add(path)
    return paths


def fully_cached_paths_v2(url, key, item_ids=None):
    if item_ids == set():
        return set()
    candidates = {}
    mount = os.environ.get('INFINIDYSK_MOUNT_PATH', '/mnt/remote/infinidysk').rstrip('/')
    library = os.environ.get('INFINIDYSK_LIBRARY_PATH', '/mnt/plex')
    with requests.Session() as client:
        client.headers['x-api-key'] = key
        status = _get(client, url, '/api/v1/native-cache')
        if (status.get('configured') is not True or status.get('available') is not True
                or status.get('recovery_mode') is not False or status.get('reason') is not None):
            return set()
        native = _get(client, url, '/api/v1/native-cache/diagnostics')
        legacy = _get(client, url, '/api/v1/native-cache/legacy/diagnostics', optional=True)
        if not isinstance(native, list) or (legacy is not None and not isinstance(legacy, list)):
            raise ValueError('Invalid cache diagnostics')
        # Coverage can merge both stores. If any backing store is unavailable,
        # do not infer readable full coverage from catalogue metadata alone.
        if any(not isinstance(f, dict) or f.get('identity_ok') is not True for f in native + (legacy or [])):
            return set()
        native_folders = {f['folder_id'] for f in native}
        legacy_folders = {f['id'] for f in legacy or []}
        for offset in range(0, 150200, 200):
            entries = _get(client, url, '/api/v1/native-cache/files', offset=offset, limit=200)
            if not isinstance(entries, list):
                raise ValueError('Invalid native cache page')
            for entry in entries:
                size = entry.get('size_bytes')
                if (entry.get('folder_id') in native_folders and type(size) is int and size > 0
                        and entry.get('cached_bytes') == size and entry.get('revision')):
                    candidates.setdefault(str(UUID(entry['item'])), set()).add((size, entry['revision']))
            if len(entries) < 200:
                break
        else:
            raise ValueError('Native cache catalogue exceeds bound')
        after, seen = '', set()
        for _ in range(10000):
            page = _get(client, url, '/api/v1/native-cache/legacy', optional=True, after=after, coverage='complete')
            if page is None:
                break
            if not isinstance(page, dict) or not isinstance(page.get('entries'), list):
                raise ValueError('Invalid original cache page')
            if any(f['id'] not in legacy_folders for f in page['folders']):
                return set()
            for entry in page['entries']:
                size = entry.get('size')
                if (entry.get('source_present') is True and entry.get('folder') in legacy_folders
                        and type(size) is int and size > 0 and entry.get('cached_bytes') == size
                        and entry.get('pending_bytes') == 0):
                    candidates.setdefault(str(UUID(entry['item'])), set()).add((size, None))
            after = page.get('next')
            if not after:
                break
            if not isinstance(after, str) or after in seen:
                raise ValueError('Repeated original cache cursor')
            seen.add(after)
        else:
            raise ValueError('Original cache catalogue exceeds bound')

    def resolve_batch(entries):
        paths = set()
        with requests.Session() as client:
            client.headers['x-api-key'] = key
            for item_id, sources in entries:
                details = _get(client, url, '/api/v1/files/' + item_id, optional=True)
                if details is not None:
                    paths.update(_verified_paths(details, item_id, sources, mount, library))
        return paths

    entries = [(item_id, sources) for item_id, sources in candidates.items()
               if item_ids is None or item_id in item_ids]
    if len(entries) <= 8:
        return resolve_batch(entries)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(resolve_batch, [entries[i::8] for i in range(8)]))
    return set().union(*results)
