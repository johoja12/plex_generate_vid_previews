# Watch-Aware Priority Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace boolean-only automatic priority with an explainable priority score based on per-user next-up prediction, selected Plex hubs, and existing manual queue intent.

**Architecture:** Add a focused `plex_generate_previews/web/priority.py` module that computes `PriorityInfo` records from Plex DB rows and PlexAPI hub/on-deck objects. Persist score/reasons on `MediaItem`, keep `is_priority` as a compatibility flag, and update scheduler/API ordering to use `priority_score`.

**Tech Stack:** Python 3, SQLModel, SQLite, PlexAPI, pytest.

---

## File Structure

- Create `plex_generate_previews/web/priority.py`: pure priority scoring helpers plus Plex DB and PlexAPI adapters.
- Create `tests/test_priority.py`: focused unit tests for scoring, hub filtering, and per-user next-up behavior.
- Modify `plex_generate_previews/web/models.py`: add `priority_score`, `priority_reasons`, `priority_last_calculated_at`.
- Modify `plex_generate_previews/web/database.py`: add SQLite migrations for new fields.
- Modify `plex_generate_previews/web/scheduler.py`: replace old priority-set path with priority-info mapping, persist priority fields, order queues by score, clear completed priority metadata.
- Modify `plex_generate_previews/web/main.py`: expose priority score/reasons and sort processing items by score.
- Modify or add scheduler tests in `tests/test_scheduler_fix.py`.

## Task 1: Priority Scoring Core

**Files:**
- Create: `plex_generate_previews/web/priority.py`
- Test: `tests/test_priority.py`

- [ ] **Step 1: Write failing tests for pure scoring**

Add tests that import missing symbols:

```python
from datetime import datetime, timedelta, timezone

from plex_generate_previews.web.priority import (
    HUB_DIRECT_SCORE,
    NEXT_EPISODE_SCORES,
    PriorityInfo,
    add_reason,
    is_included_hub,
    is_excluded_hub,
    recency_multiplier,
)


def test_add_reason_caps_score_and_keeps_reasons():
    info = PriorityInfo()

    add_reason(info, "next_episode", 800, account_id=1)
    add_reason(info, "hub", 300, hub="Trending")

    assert info.score == 999
    assert info.reasons == [
        {"type": "next_episode", "score": 800, "account_id": 1},
        {"type": "hub", "score": 300, "hub": "Trending"},
    ]


def test_recency_multiplier_decays_and_ignores_old_views():
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)

    assert recency_multiplier(now - timedelta(days=1), now) == 1.0
    assert recency_multiplier(now - timedelta(days=4), now) == 0.8
    assert recency_multiplier(now - timedelta(days=20), now) == 0.5
    assert recency_multiplier(now - timedelta(days=31), now) == 0.0


def test_hub_name_filters_include_only_requested_hubs():
    assert is_included_hub("Trending")
    assert is_included_hub("Popular TV This Year")
    assert is_included_hub("Most Watched This Week")
    assert is_included_hub("Favourite TV")
    assert is_included_hub("Favorite Movies")
    assert not is_included_hub("Start Watching")
    assert not is_included_hub("Rediscover")
    assert not is_included_hub("Top Unwatched Movies")


def test_recency_hubs_are_excluded_even_if_other_terms_match():
    assert is_excluded_hub("Recently Added in Movies")
    assert is_excluded_hub("Recently Released Episodes")
    assert not is_included_hub("Recently Added Popular Movies")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python3 -m pytest tests/test_priority.py -q`

Expected: FAIL because `plex_generate_previews.web.priority` does not exist.

- [ ] **Step 3: Implement minimal scoring core**

Create `priority.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

MAX_SCORE = 999
ON_DECK_SCORE = 900
NEXT_EPISODE_SCORES = (800, 760, 720)
MULTI_USER_BOOST = 75
MULTI_USER_BOOST_CAP = 225
HUB_DIRECT_SCORE = 300
HUB_SHOW_SCORE = 250
RECENT_WATCHED_SCORE = 100


@dataclass
class PriorityInfo:
    score: int = 0
    reasons: list[dict[str, Any]] = field(default_factory=list)


def add_reason(info: PriorityInfo, reason_type: str, score: int, **metadata: Any) -> None:
    reason = {"type": reason_type, "score": score}
    reason.update({key: value for key, value in metadata.items() if value is not None})
    info.reasons.append(reason)
    info.score = min(MAX_SCORE, info.score + score)


def recency_multiplier(viewed_at: datetime, now: datetime) -> float:
    age_days = (now - viewed_at).total_seconds() / 86400
    if age_days <= 2:
        return 1.0
    if age_days <= 7:
        return 0.8
    if age_days <= 30:
        return 0.5
    return 0.0


def is_excluded_hub(title: str) -> bool:
    lowered = title.lower()
    return "recently added" in lowered or "recently released" in lowered


def is_included_hub(title: str) -> bool:
    if is_excluded_hub(title):
        return False
    lowered = title.lower()
    return any(term in lowered for term in ("trending", "popular", "most watched", "favorite", "favourite"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python3 -m pytest tests/test_priority.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add plex_generate_previews/web/priority.py tests/test_priority.py
git commit -m "Add priority scoring primitives"
```

## Task 2: Per-User Next-Up Prediction

**Files:**
- Modify: `plex_generate_previews/web/priority.py`
- Test: `tests/test_priority.py`

- [ ] **Step 1: Write failing tests for next-3 prediction**

Append:

```python
from plex_generate_previews.web.priority import EpisodeRow, WatchEvent, score_next_up_episodes


def test_next_three_unwatched_episodes_are_scored_per_user():
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    episodes = [
        EpisodeRow(101, 10, 1, 1),
        EpisodeRow(102, 10, 1, 2),
        EpisodeRow(103, 10, 1, 3),
        EpisodeRow(104, 10, 1, 4),
        EpisodeRow(105, 10, 1, 5),
    ]
    watches = [WatchEvent(account_id=1, rating_key=102, show_id=10, season_index=1, episode_index=2, viewed_at=now)]

    result = score_next_up_episodes(watches, episodes, missing_rating_keys={103, 104, 105}, now=now)

    assert [result[key].score for key in (103, 104, 105)] == [800, 760, 720]
    assert 101 not in result
    assert 102 not in result


def test_different_users_produce_independent_next_up_candidates():
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    episodes = [
        EpisodeRow(201, 20, 1, 1),
        EpisodeRow(202, 20, 1, 2),
        EpisodeRow(203, 20, 1, 3),
        EpisodeRow(301, 30, 1, 1),
        EpisodeRow(302, 30, 1, 2),
    ]
    watches = [
        WatchEvent(account_id=1, rating_key=201, show_id=20, season_index=1, episode_index=1, viewed_at=now),
        WatchEvent(account_id=2, rating_key=301, show_id=30, season_index=1, episode_index=1, viewed_at=now),
    ]

    result = score_next_up_episodes(watches, episodes, missing_rating_keys={202, 203, 302}, now=now)

    assert 202 in result
    assert 203 in result
    assert 302 in result


def test_shared_next_up_candidate_gets_overlap_boost():
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    episodes = [EpisodeRow(401, 40, 1, 1), EpisodeRow(402, 40, 1, 2)]
    watches = [
        WatchEvent(account_id=1, rating_key=401, show_id=40, season_index=1, episode_index=1, viewed_at=now),
        WatchEvent(account_id=2, rating_key=401, show_id=40, season_index=1, episode_index=1, viewed_at=now),
    ]

    result = score_next_up_episodes(watches, episodes, missing_rating_keys={402}, now=now)

    assert result[402].score == 875
    assert [reason["type"] for reason in result[402].reasons] == ["next_episode", "multi_user_overlap"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python3 -m pytest tests/test_priority.py -q`

Expected: FAIL because `EpisodeRow`, `WatchEvent`, and `score_next_up_episodes` are missing.

- [ ] **Step 3: Implement next-up scoring**

Add dataclasses and `score_next_up_episodes()` to `priority.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python3 -m pytest tests/test_priority.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add plex_generate_previews/web/priority.py tests/test_priority.py
git commit -m "Score per-user next-up episodes"
```

## Task 3: Hub Candidate Scoring

**Files:**
- Modify: `plex_generate_previews/web/priority.py`
- Test: `tests/test_priority.py`

- [ ] **Step 1: Write failing tests for hub scoring**

Add tests for direct movie/episode IDs, show IDs mapping to first 3 missing episodes, season IDs mapping to first 3 missing season episodes, and broad recency hubs ignored.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python3 -m pytest tests/test_priority.py -q`

Expected: FAIL because hub scoring function is missing.

- [ ] **Step 3: Implement hub scoring helpers**

Add a function such as:

```python
def score_hub_items(hub_items, episode_rows, missing_rating_keys):
    result = {}
    # Ignore excluded hub names before processing items.
    # Add HUB_DIRECT_SCORE for movie/episode items whose rating keys are missing.
    # For show items, add HUB_SHOW_SCORE to first 3 missing episodes in that show.
    # For season items, add HUB_SHOW_SCORE to first 3 missing episodes in that season.
    return result
```

Use item `type` values `movie`, `episode`, `show`, and `season`. Apply `HUB_DIRECT_SCORE` for direct movie/episode matches and `HUB_SHOW_SCORE` to first 3 missing episodes for show/season hub items.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python3 -m pytest tests/test_priority.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add plex_generate_previews/web/priority.py tests/test_priority.py
git commit -m "Score selected Plex hub candidates"
```

## Task 4: Persist Priority Fields

**Files:**
- Modify: `plex_generate_previews/web/models.py`
- Modify: `plex_generate_previews/web/database.py`
- Test: `tests/test_scheduler_fix.py`

- [ ] **Step 1: Write failing model/migration test**

Add a test that creates tables/migrations and asserts new `mediaitem` columns exist.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python3 -m pytest tests/test_scheduler_fix.py -q`

Expected: FAIL because columns are missing.

- [ ] **Step 3: Add fields and migrations**

Add `priority_score`, `priority_reasons`, and `priority_last_calculated_at` to `MediaItem`; add SQLite `ALTER TABLE` migrations.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python3 -m pytest tests/test_scheduler_fix.py tests/test_priority.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add plex_generate_previews/web/models.py plex_generate_previews/web/database.py tests/test_scheduler_fix.py
git commit -m "Persist priority score metadata"
```

## Task 5: Scheduler Integration and Ordering

**Files:**
- Modify: `plex_generate_previews/web/scheduler.py`
- Modify: `plex_generate_previews/web/priority.py`
- Test: `tests/test_scheduler_fix.py`

- [ ] **Step 1: Write failing scheduler tests**

Add tests for `_apply_priority_info_to_item()`, queue ordering by `priority_score`, manual `QUEUED` override, and completed-item priority cleanup.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python3 -m pytest tests/test_scheduler_fix.py -q`

Expected: FAIL because scheduler does not use priority scores.

- [ ] **Step 3: Implement scheduler changes**

Replace `priority_item_keys` with `priority_info_by_key`, persist score/reasons/timestamp, order fetch queries by `MediaItem.priority_score.desc()`, and clear priority fields for completed items.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python3 -m pytest tests/test_scheduler_fix.py tests/test_priority.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add plex_generate_previews/web/scheduler.py plex_generate_previews/web/priority.py tests/test_scheduler_fix.py
git commit -m "Use priority scores in scheduler"
```

## Task 6: API Exposure

**Files:**
- Modify: `plex_generate_previews/web/main.py`
- Test: `tests/test_scheduler_fix.py` or new API-focused test if existing fixtures support it.

- [ ] **Step 1: Write failing API serialization test**

Test that item payloads include `priority_score` and `priority_reasons`, and processing ordering uses `priority_score`.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python3 -m pytest tests/test_scheduler_fix.py -q`

Expected: FAIL because API does not include new fields.

- [ ] **Step 3: Add API fields and processing ordering**

Include priority fields wherever `is_priority` is emitted and replace processing-item priority sort with `priority_score.desc()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python3 -m pytest tests/test_scheduler_fix.py tests/test_priority.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add plex_generate_previews/web/main.py tests/test_scheduler_fix.py
git commit -m "Expose priority score in API"
```

## Task 7: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused tests**

Run: `PYTHONPATH=. python3 -m pytest tests/test_priority.py tests/test_scheduler_fix.py -q`

Expected: PASS.

- [ ] **Step 2: Run broader impacted tests**

Run: `PYTHONPATH=. python3 -m pytest tests/test_worker.py tests/test_media_processing.py tests/test_scheduler_fix.py tests/test_priority.py -q`

Expected: PASS or report pre-existing unrelated failures with exact output.

- [ ] **Step 3: Run a read-only production data smoke check**

Run a script that imports the priority module, connects to the configured Plex DB and Plex API, computes priority candidates, and prints only counts/top reason summaries.

- [ ] **Step 4: Commit any final fixes**

Commit only implementation files and tests.
