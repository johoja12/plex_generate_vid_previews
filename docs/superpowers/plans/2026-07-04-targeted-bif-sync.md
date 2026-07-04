# Targeted BIF Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make library sync avoid the full Plex `Media/localhost` bundle-tree glob and check BIF existence only for bundle hashes present in the current Plex library, while keeping the full scan available as an explicit Settings maintenance tool.

**Architecture:** Keep sync behavior in `plex_generate_previews/web/scheduler.py`, but replace the sync-time full filesystem scan with a targeted BIF map builder that derives exact `index-sd.bif` paths from known bundle hashes. Preserve `_scan_filesystem_for_bifs()` for maintenance/debug callers and existing tests, but remove it from the normal `sync_library()` hot path.

**Tech Stack:** Python, FastAPI scheduler thread, SQLite/SQLModel, Plex config filesystem layout, pytest.

---

## File Structure

- Modify: `plex_generate_previews/web/scheduler.py`
  - Add a focused helper that accepts bundle hashes and returns the same `{full_bundle_hash: bif_path}` shape as `_scan_filesystem_for_bifs()`.
  - Update `sync_library()` to fetch Plex items first, then build the BIF map from those item hashes instead of globbing the entire Plex bundle tree.
  - Add timing logs for the targeted BIF check.
- Modify: `plex_generate_previews/web/main.py`
  - Add an authenticated Settings endpoint that runs the preserved full BIF scan on demand and reports count, duration, and samples.
- Modify: `plex_generate_previews/web/templates/settings.html`
  - Add a Maintenance action for the full BIF scan so the expensive operation is explicit.
- Modify: `tests/test_scheduler_fix.py`
  - Add unit tests for targeted BIF map behavior with Plex's stripped bundle directory naming.
  - Add a sync-level regression test that proves `sync_library()` does not call `_scan_filesystem_for_bifs()`.
- Modify: `tests/test_template_response_signature.py`
  - Add a template assertion that the Settings page exposes the full BIF scan action.
- Optional later: `docs/superpowers/plans/2026-07-04-targeted-bif-sync.md`
  - Update this plan if implementation discoveries change scope.

## Key Design

Plex BIF paths are deterministic from a full 40-character bundle hash:

```python
first_char = bundle_hash[0]
stripped_hash = bundle_hash[1:]
bif_path = f"{plex_config_folder}/Media/localhost/{first_char}/{stripped_hash}.bundle/Contents/Indexes/index-sd.bif"
```

The sync already has every Plex item bundle hash in `all_items` and every media part bundle hash in `media_parts_map`. The new helper should check only those hashes, deduplicate them, and return entries for hashes whose BIF file exists.

---

### Task 1: Add Targeted BIF Map Helper

**Files:**
- Modify: `plex_generate_previews/web/scheduler.py`
- Test: `tests/test_scheduler_fix.py`

- [ ] **Step 1: Write failing tests for targeted BIF path checks**

Add these tests near `test_scan_filesystem_for_bifs_with_stripped_hash_bif` in `tests/test_scheduler_fix.py`:

```python
def test_build_bif_map_for_bundle_hashes_checks_only_known_hashes(temp_dir, mock_config, monkeypatch):
    plex_config_path = os.path.join(temp_dir, "plex_config")
    mock_config.plex_config_folder = plex_config_path

    existing_hash = "98ee3ecdf4aba34b7708a7d8c92422f4e96eebf2"
    missing_hash = "e730b9ff1874a14b1e4f09f671a1f278531d04fb"
    existing_bif_path = os.path.join(
        plex_config_path,
        "Media",
        "localhost",
        existing_hash[0],
        f"{existing_hash[1:]}.bundle",
        "Contents",
        "Indexes",
        "index-sd.bif",
    )
    os.makedirs(os.path.dirname(existing_bif_path), exist_ok=True)
    with open(existing_bif_path, "w") as f:
        f.write("dummy bif content")

    checked_paths = []
    real_exists = os.path.exists

    def tracking_exists(path):
        checked_paths.append(path)
        return real_exists(path)

    monkeypatch.setattr(scheduler_module.os.path, "exists", tracking_exists)

    scheduler = Scheduler()
    scheduler.config = mock_config

    result = scheduler._build_bif_map_for_bundle_hashes(
        [existing_hash, missing_hash, existing_hash, None, ""]
    )

    assert result == {existing_hash: existing_bif_path}
    assert checked_paths == [existing_bif_path, scheduler._get_bif_path(missing_hash)]
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
pytest tests/test_scheduler_fix.py::test_build_bif_map_for_bundle_hashes_checks_only_known_hashes -v
```

Expected: FAIL with `AttributeError: 'Scheduler' object has no attribute '_build_bif_map_for_bundle_hashes'`.

- [ ] **Step 3: Implement the helper**

Add this method immediately after `_scan_filesystem_for_bifs()` in `plex_generate_previews/web/scheduler.py`:

```python
    def _build_bif_map_for_bundle_hashes(self, bundle_hashes):
        """
        Build a BIF map by checking exact expected BIF paths for known Plex bundle hashes.

        This avoids walking the entire Plex Media/localhost bundle tree over NFS during sync.
        Returns the same shape as _scan_filesystem_for_bifs(): {bundle_hash: bif_path}.
        """
        if not self.config:
            logger.error("Cannot build BIF map: scheduler not configured")
            return {}

        bundle_hash_map = {}
        seen_hashes = set()

        for raw_hash in bundle_hashes:
            if not raw_hash:
                continue

            bundle_hash = str(raw_hash).lower()
            if bundle_hash in seen_hashes:
                continue
            seen_hashes.add(bundle_hash)

            bif_path = self._get_bif_path(bundle_hash)
            if os.path.exists(bif_path):
                bundle_hash_map[bundle_hash] = bif_path

        logger.debug(
            f"Targeted BIF check found {len(bundle_hash_map)} existing BIF files "
            f"from {len(seen_hashes)} bundle hashes"
        )
        return bundle_hash_map
```

- [ ] **Step 4: Run the test and verify it passes**

Run:

```bash
pytest tests/test_scheduler_fix.py::test_build_bif_map_for_bundle_hashes_checks_only_known_hashes -v
```

Expected: PASS.

- [ ] **Step 5: Run existing BIF scan test**

Run:

```bash
pytest tests/test_scheduler_fix.py::test_scan_filesystem_for_bifs_with_stripped_hash_bif -v
```

Expected: PASS. This confirms the debug/full scan helper still works.

---

### Task 2: Move Sync BIF Detection After Plex Item Fetch

**Files:**
- Modify: `plex_generate_previews/web/scheduler.py`
- Test: `tests/test_scheduler_fix.py`

- [ ] **Step 1: Write sync regression test**

Add this test to `tests/test_scheduler_fix.py`:

```python
def test_sync_library_uses_targeted_bif_checks_not_full_scan(mock_config, monkeypatch):
    scheduler = Scheduler()
    scheduler.config = mock_config
    scheduler.sync_in_progress = False

    class FakePlex:
        pass

    class FakeSection:
        title = "Movies"

    item_key = 123
    bundle_hash = "98ee3ecdf4aba34b7708a7d8c92422f4e96eebf2"
    added_at = datetime(2026, 1, 1)

    monkeypatch.setattr(scheduler_module, "plex_server", lambda config: FakePlex())
    monkeypatch.setattr(
        scheduler_module,
        "get_library_sections",
        lambda plex, config, database_only=True: [
            (FakeSection(), [(item_key, "Movie", "movie", bundle_hash, added_at)])
        ],
    )
    monkeypatch.setattr(
        scheduler_module,
        "get_all_media_parts_batch",
        lambda plex_config_folder, rating_keys: {
            item_key: [("/mnt/plex/movie.mkv", bundle_hash)]
        },
    )
    monkeypatch.setattr(Scheduler, "validate_mount_paths", lambda self: (True, []))
    monkeypatch.setattr(Scheduler, "detect_priority_items", lambda self, plex, missing_rating_keys=None: {})
    monkeypatch.setattr(Scheduler, "_count_newly_media_missing_items", lambda self, session: (0, 1))
    monkeypatch.setattr(scheduler_module.os.path, "exists", lambda path: True)

    full_scan_called = False

    def fail_full_scan(self):
        nonlocal full_scan_called
        full_scan_called = True
        raise AssertionError("_scan_filesystem_for_bifs should not run during sync_library")

    targeted_calls = []

    def fake_targeted_scan(self, bundle_hashes):
        targeted_calls.append(list(bundle_hashes))
        return {bundle_hash: self._get_bif_path(bundle_hash)}

    monkeypatch.setattr(Scheduler, "_scan_filesystem_for_bifs", fail_full_scan)
    monkeypatch.setattr(Scheduler, "_build_bif_map_for_bundle_hashes", fake_targeted_scan)

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(scheduler_module, "engine", engine)

    scheduler.sync_library()

    assert full_scan_called is False
    assert targeted_calls == [[bundle_hash]]

    with Session(engine) as session:
        item = session.get(MediaItem, item_key)
        settings = session.get(scheduler_module.AppSettings, 1)

    assert item is not None
    assert item.status == PreviewStatus.COMPLETED
    assert settings.last_sync_summary == (
        '{"movies_added": 1, "movies_updated": 0, "movies_deleted": 0, '
        '"episodes_added": 0, "episodes_updated": 0, "episodes_deleted": 0}'
    )
```

- [ ] **Step 2: Run the regression test and verify it fails**

Run:

```bash
pytest tests/test_scheduler_fix.py::test_sync_library_uses_targeted_bif_checks_not_full_scan -v
```

Expected: FAIL because `sync_library()` still calls `_scan_filesystem_for_bifs()`.

- [ ] **Step 3: Refactor `sync_library()` phase order**

In `plex_generate_previews/web/scheduler.py`, replace the existing Step 1/Step 2 ordering in `sync_library()`:

```python
            # Step 1: Scan filesystem once for all BIF files
            logger.info("Scanning filesystem for BIF files...")
            bif_scan_start = time.time()
            bundle_hash_map = self._scan_filesystem_for_bifs()
            logger.info(f"Found {len(bundle_hash_map)} existing BIF files in {time.time() - bif_scan_start:.2f}s")

            # Step 2: Get items from Plex
            logger.info("Fetching items from Plex...")
            plex_fetch_start = time.time()
            plex = plex_server(self.config)
```

with:

```python
            # Step 1: Get items from Plex
            logger.info("Fetching items from Plex...")
            plex_fetch_start = time.time()
            plex = plex_server(self.config)
```

Then, after:

```python
            logger.info(f"Fetched {len(all_items)} items from Plex in {time.time() - plex_fetch_start:.2f}s")
```

insert:

```python
            # Step 2: Check exact expected BIF paths only for bundle hashes in this Plex library.
            logger.info("Checking BIF files for current Plex items...")
            bif_scan_start = time.time()
            item_bundle_hashes = [
                bundle_hash
                for _, _, _, _, bundle_hash, _ in all_items
                if bundle_hash
            ]
            bundle_hash_map = self._build_bif_map_for_bundle_hashes(item_bundle_hashes)
            logger.info(
                f"Found {len(bundle_hash_map)} existing BIF files from "
                f"{len(set(item_bundle_hashes))} bundle hashes in {time.time() - bif_scan_start:.2f}s"
            )
```

- [ ] **Step 4: Run the regression test and verify it passes**

Run:

```bash
pytest tests/test_scheduler_fix.py::test_sync_library_uses_targeted_bif_checks_not_full_scan -v
```

Expected: PASS.

---

### Task 3: Include Multi-Part Bundle Hashes In Targeted BIF Checks

**Files:**
- Modify: `plex_generate_previews/web/scheduler.py`
- Test: `tests/test_scheduler_fix.py`

- [ ] **Step 1: Write multi-part regression test**

Add this test:

```python
def test_sync_library_targeted_bif_checks_include_media_part_hashes(mock_config, monkeypatch):
    scheduler = Scheduler()
    scheduler.config = mock_config

    class FakePlex:
        pass

    class FakeSection:
        title = "Movies"

    item_key = 123
    primary_hash = "98ee3ecdf4aba34b7708a7d8c92422f4e96eebf2"
    part_hash = "e730b9ff1874a14b1e4f09f671a1f278531d04fb"

    monkeypatch.setattr(scheduler_module, "plex_server", lambda config: FakePlex())
    monkeypatch.setattr(
        scheduler_module,
        "get_library_sections",
        lambda plex, config, database_only=True: [
            (FakeSection(), [(item_key, "Movie", "movie", primary_hash, datetime(2026, 1, 1))])
        ],
    )
    monkeypatch.setattr(
        scheduler_module,
        "get_all_media_parts_batch",
        lambda plex_config_folder, rating_keys: {
            item_key: [
                ("/mnt/plex/movie-1080p.mkv", primary_hash),
                ("/mnt/plex/movie-4k.mkv", part_hash),
            ]
        },
    )
    monkeypatch.setattr(Scheduler, "validate_mount_paths", lambda self: (True, []))
    monkeypatch.setattr(Scheduler, "detect_priority_items", lambda self, plex, missing_rating_keys=None: {})
    monkeypatch.setattr(Scheduler, "_count_newly_media_missing_items", lambda self, session: (0, 1))
    monkeypatch.setattr(scheduler_module.os.path, "exists", lambda path: True)

    targeted_calls = []

    def fake_targeted_scan(self, bundle_hashes):
        targeted_calls.append(list(bundle_hashes))
        return {
            primary_hash: self._get_bif_path(primary_hash),
            part_hash: self._get_bif_path(part_hash),
        }

    monkeypatch.setattr(Scheduler, "_build_bif_map_for_bundle_hashes", fake_targeted_scan)

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(scheduler_module, "engine", engine)

    scheduler.sync_library()

    assert targeted_calls == [[primary_hash, part_hash]]

    with Session(engine) as session:
        item = session.get(MediaItem, item_key)

    assert item.status == PreviewStatus.COMPLETED
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
pytest tests/test_scheduler_fix.py::test_sync_library_targeted_bif_checks_include_media_part_hashes -v
```

Expected: FAIL because the initial targeted hash list includes only top-level item bundle hashes.

- [ ] **Step 3: Move targeted BIF map construction after media part query**

In `sync_library()`, remove the targeted BIF block inserted immediately after `Fetched ... items from Plex`.

After this existing block:

```python
            logger.info("Batch querying media parts from Plex database...")
            batch_query_start = time.time()
            all_rating_keys = [int(item[1]) for item in all_items]
            media_parts_map = get_all_media_parts_batch(self.config.plex_config_folder, all_rating_keys)
            logger.info(f"Queried media parts for {len(all_rating_keys)} items in {time.time() - batch_query_start:.2f}s")
```

insert:

```python
            # Step 4: Check exact expected BIF paths only for bundle hashes in this Plex library.
            logger.info("Checking BIF files for current Plex items...")
            bif_scan_start = time.time()
            bundle_hashes_to_check = []
            bundle_hashes_to_check.extend(
                bundle_hash
                for _, _, _, _, bundle_hash, _ in all_items
                if bundle_hash
            )
            for media_parts in media_parts_map.values():
                bundle_hashes_to_check.extend(
                    bundle_hash
                    for _, bundle_hash in media_parts
                    if bundle_hash
                )
            unique_bundle_hash_count = len({str(bundle_hash).lower() for bundle_hash in bundle_hashes_to_check})
            bundle_hash_map = self._build_bif_map_for_bundle_hashes(bundle_hashes_to_check)
            logger.info(
                f"Found {len(bundle_hash_map)} existing BIF files from "
                f"{unique_bundle_hash_count} bundle hashes in {time.time() - bif_scan_start:.2f}s"
            )
```

Renumber comments below this point only if they are misleading; avoid unrelated cleanup.

- [ ] **Step 4: Run both sync regression tests**

Run:

```bash
pytest tests/test_scheduler_fix.py::test_sync_library_uses_targeted_bif_checks_not_full_scan tests/test_scheduler_fix.py::test_sync_library_targeted_bif_checks_include_media_part_hashes -v
```

Expected: both PASS.

---

### Task 4: Settings Full BIF Scan Maintenance Tool

**Files:**
- Modify: `plex_generate_previews/web/main.py`
- Modify: `plex_generate_previews/web/templates/settings.html`
- Test: `tests/test_scheduler_fix.py`
- Test: `tests/test_template_response_signature.py`

- [x] **Step 1: Add endpoint regression test**

Added a test that monkeypatches `main.scheduler._scan_filesystem_for_bifs()` and verifies `scan_all_bifs()` returns `found`, `duration_seconds`, and sample hashes.

- [x] **Step 2: Implement backend endpoint**

Added `POST /api/settings/scan-all-bifs` in `plex_generate_previews/web/main.py`. The endpoint requires login, checks `scheduler.config`, runs `_scan_filesystem_for_bifs()`, and returns scan metadata without changing database state.

- [x] **Step 3: Add Settings UI action**

Added a `Full BIF Scan` row in the Maintenance card of `settings.html`, with Alpine state `scanningAllBifs`, `fullBifScanResult`, and method `scanAllBifs()`.

- [x] **Step 4: Add template regression test**

Asserted the Settings template includes `Full BIF Scan`, `scanAllBifs`, and `/api/settings/scan-all-bifs`.

- [x] **Step 5: Run focused tests**

Run:

```bash
pytest tests/test_scheduler_fix.py -v
pytest tests/test_template_response_signature.py -v
```

Expected: both pass.

---

### Task 5: Verification And Deployment Readiness

**Files:**
- No code edits unless tests reveal an issue.

- [ ] **Step 1: Run focused scheduler tests**

Run:

```bash
pytest tests/test_scheduler_fix.py -v
```

Expected: all tests in `tests/test_scheduler_fix.py` pass.

- [ ] **Step 2: Run full suite**

Run:

```bash
pytest -q
```

Expected: full suite passes.

- [ ] **Step 3: Run syntax and whitespace checks**

Run:

```bash
python3 -m compileall -q plex_generate_previews tests
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 4: Inspect diff for scope**

Run:

```bash
git diff -- plex_generate_previews/web/scheduler.py tests/test_scheduler_fix.py
```

Expected: diff only adds targeted BIF checking, updates sync order, and adds tests. It must not change queue priority ordering, media processing, auth, or unrelated UI behavior.

- [ ] **Step 5: Commit**

Run:

```bash
git add plex_generate_previews/web/scheduler.py tests/test_scheduler_fix.py docs/superpowers/plans/2026-07-04-targeted-bif-sync.md
git commit -m "Speed up sync BIF checks"
```

Expected: commit succeeds.

---

### Task 6: Production Smoke After Deploy

**Files:**
- No source edits.

- [ ] **Step 1: Build image**

Run from repo root:

```bash
docker compose build --no-cache
```

Expected: image `plex-generate-previews:latest` builds successfully.

- [ ] **Step 2: Restart service**

Run:

```bash
cd /opt/docker/plex-previews && docker compose up -d
```

Expected: `plex-previews` recreates and starts.

- [ ] **Step 3: Verify service health**

Run:

```bash
docker ps --filter name=plex-previews --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
curl -fsS -o /tmp/plexpreview_health.html -w '%{http_code}\n' http://127.0.0.1:8008/
```

Expected: container is `Up`; HTTP code is `307` when unauthenticated.

- [ ] **Step 4: Trigger sync and verify BIF phase timing**

Use the web UI Sync button, then run:

```bash
docker logs plex-previews --since 10m --timestamps 2>&1 \
  | sed -r 's/\x1B\[[0-9;]*[mK]//g' \
  | grep -E 'Checking BIF files|Found [0-9]+ existing BIF files from|Library sync complete in'
```

Expected: logs include `Checking BIF files for current Plex items...`; the BIF phase should be seconds or low minutes, not the previous ~1092 seconds.

- [ ] **Step 5: Push**

Run:

```bash
git push origin main
```

Expected: push succeeds.

---

## Self-Review

- Spec coverage: The plan addresses the observed sync bottleneck, preserves existing full-scan behavior as an explicit Settings maintenance action, includes multi-part media hashes, and defines production timing verification.
- Placeholder scan: No `TBD`, generic "add tests", or unspecified validation steps remain.
- Type consistency: New helper is `_build_bif_map_for_bundle_hashes(self, bundle_hashes)` and every task uses that exact name; map shape remains `{bundle_hash: bif_path}`.
