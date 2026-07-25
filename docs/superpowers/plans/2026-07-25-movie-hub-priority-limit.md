# Movie Hub Priority Limit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give hub-derived priority only to the first 20 movies in each Plex hub while leaving later movies in the normal queue and preserving non-movie hub behavior.

**Architecture:** Keep the existing hub collection and downstream scoring pipeline. Add a per-hub movie counter while converting Plex hub items, skipping movies after the twentieth but continuing through the existing 100-item scan so non-movie entries remain available for their current scoring behavior.

**Tech Stack:** Python, PlexAPI-compatible objects, pytest, `unittest.mock.MagicMock`

---

### Task 1: Limit hub-derived movie priority

**Files:**
- Modify: `plex_generate_previews/web/scheduler.py:1644-1698`
- Test: `tests/test_scheduler_fix.py`

- [ ] **Step 1: Write the failing collector regression test**

Add this test to `tests/test_scheduler_fix.py`:

```python
def test_collect_priority_hub_items_keeps_only_first_twenty_movies():
    scheduler = Scheduler()
    movies = [
        MagicMock(ratingKey=str(index), type="movie", title=f"Movie {index}")
        for index in range(1, 26)
    ]
    show = MagicMock(ratingKey="1000", type="show", title="TV Show")
    hub = MagicMock(title="Trending", items=movies + [show])
    section = MagicMock()
    section.hubs.return_value = [hub]
    plex = MagicMock()
    plex.library.sections.return_value = [section]

    result = scheduler._collect_priority_hub_items(plex)

    assert [item.rating_key for item in result["Trending"] if item.item_type == "movie"] == list(
        range(1, 21)
    )
    assert [(item.rating_key, item.item_type) for item in result["Trending"] if item.item_type != "movie"] == [
        (1000, "show")
    ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/test_scheduler_fix.py::test_collect_priority_hub_items_keeps_only_first_twenty_movies -v
```

Expected: FAIL because all 25 movie entries are currently collected.

- [ ] **Step 3: Implement the per-hub movie limit**

In `Scheduler._collect_priority_hub_items`, initialize a movie counter for each
hub and skip movie entries after the first 20:

```python
converted_items = []
movie_count = 0
for item in raw_items[:100]:
    try:
        rating_key = int(getattr(item, "ratingKey"))
    except (TypeError, ValueError):
        continue

    item_type = getattr(item, "type", "")
    if item_type == "movie":
        if movie_count >= 20:
            continue
        movie_count += 1
```

Leave the existing season conversion and `HubItem` construction unchanged.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_scheduler_fix.py::test_collect_priority_hub_items_keeps_only_first_twenty_movies tests/test_priority.py -v
```

Expected: all selected tests PASS.

- [ ] **Step 5: Run the full suite**

Run:

```bash
pytest
```

Expected: all tests PASS with no new errors or warnings caused by this change.

- [ ] **Step 6: Commit the implementation**

```bash
git add plex_generate_previews/web/scheduler.py tests/test_scheduler_fix.py docs/superpowers/plans/2026-07-25-movie-hub-priority-limit.md
git commit -m "Limit movie hub priority to top 20"
```
