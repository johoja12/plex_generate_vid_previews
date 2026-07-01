# Watch-Aware Priority Queue Design

## Context

Issue: https://github.com/johoja12/plex_generate_vid_previews/issues/1

The current scheduler has a boolean `MediaItem.is_priority` flag. During sync, priority is detected from Plex on-deck items, recent watch history, and adjacent episodes. For watched episodes, the existing logic promotes every episode in the same season. Queue selection then orders items by manual queued status, `is_priority`, `queue_order`, and `updated_at`.

Production data shows this creates a broad priority bucket: hundreds of missing items can be priority at once, including many episodes from the same series with equal rank. This does not reflect what a viewer is most likely to watch next.

## Research Findings

The app database currently has no ranked priority fields. `mediaitem` has `queue_order` and `is_priority`, but no score, reason, or calculation timestamp.

The Plex database has enough read-only data for efficient scoring:

- `metadata_item_views` provides `account_id`, `guid`, show/season/episode display fields, `metadata_type`, and `viewed_at`.
- `metadata_items` provides rating keys, parent relationships, metadata type, title, season/episode index, and added timestamps.
- Existing indexes cover `metadata_item_views.viewed_at`, `metadata_item_views.guid`, `metadata_items.guid`, `metadata_items.parent_id`, and metadata type queries.

PlexAPI hub access was verified against the configured server:

- `LibrarySection.hubs()` returns populated hubs for movie and TV libraries.
- `LibrarySection.managedHubs()` is available, but returned hub definitions without item lists in the tested environment.
- Movie hubs such as `Trending`, `Most Watched This Week`, `Most Watched This Month`, `Most Watched This Year`, `Popular`, and `Top Unwatched Movies` return movie rating keys that map directly to queueable `MediaItem` rows.
- TV curated hubs such as `Start Watching`, `Rediscover`, `Trending TV Right Now`, and `Popular TV This Year` return show rating keys, not episode rating keys.
- Broad recency hubs such as `Recently Added`, `Recently Released Movies`, and `Recently Released Episodes` can be very large and should not receive a generic priority boost.

## Goals

- Rank automatic priority items by likely next watch, not by a broad boolean flag.
- Prioritize the next 3 unwatched episodes after a user's latest watched episode in a show/season.
- Use Plex hubs for both movies and TV, but only curated/discovery hubs, not broad recently-added or recently-released hubs.
- Preserve manual queue controls as the strongest user intent.
- Keep priority explainable in logs and API/UI data.
- Avoid large Plex API fan-out during normal sync.

## Non-Goals

- No machine learning model.
- No generic recently-added boost.
- No configurable score-weight UI in the first version.
- No replacement of existing manual queue controls.

## Data Model

Add these fields to `MediaItem`:

- `priority_score INTEGER DEFAULT 0`
- `priority_reasons TEXT` containing a JSON array of reason objects.
- `priority_last_calculated_at DATETIME`

Keep `is_priority` for compatibility. It should become a derived compatibility flag where `is_priority = priority_score > 0`.

Reason objects should be compact and explainable, for example:

```json
[
  {"type": "next_episode", "score": 700, "user_count": 1, "source": "watch_history"},
  {"type": "hub", "score": 250, "hub": "Trending"}
]
```

## Scoring Model

Manual queue actions remain above automatic scoring because `QUEUED` items continue to sort before `MISSING` items.

Initial automatic scores:

- On-deck / continue watching episode or movie: `900`
- Next 1 unwatched episode after latest watched episode: `800`
- Next 2 unwatched episode: `760`
- Next 3 unwatched episode: `720`
- Multi-user overlap boost: `+75` per additional user with recent activity for the same item/show, capped at `+225`
- Curated Plex hub movie or direct episode: `300`
- Curated Plex hub show: `250` applied to the first 3 missing episodes for that show
- Recent watched item itself: `100`, mainly for visibility and compatibility, not enough to outrank next episodes

Scores are additive, capped at `999`.

Recency should affect watch-history-derived signals. Start with a simple decay:

- viewed within 2 days: full score
- viewed within 7 days: 80 percent
- viewed within 30 days: 50 percent
- older than 30 days: ignore for predictive scoring

## Hub Selection

Include curated/discovery hub names matching these terms:

- `trending`
- `popular`
- `most watched`
- `start watching`
- `rediscover`
- `top unwatched`
- `favorite`
- `favourite`

Exclude broad recency hub names matching:

- `recently added`
- `recently released`

Hub scoring should read from `section.hubs()` only. `managedHubs()` can be left for future diagnostics because it did not return populated item lists in testing.

For movie libraries, hub item rating keys map directly to `MediaItem.id`.

For TV libraries:

- Episode hub items map directly to `MediaItem.id`.
- Show hub items should not promote an entire show. They should promote only the first 3 missing queueable episodes for that show, ordered by season and episode index.
- Season hub items should promote only the first 3 missing queueable episodes in that season.

## Priority Calculation Flow

Create a focused priority calculation component instead of growing `Scheduler` further. A module such as `plex_generate_previews/web/priority.py` should own:

- Reading watch history and candidate episode relationships from the Plex DB.
- Reading curated Plex hub candidates through PlexAPI.
- Combining reasons and scores.
- Returning a mapping of `rating_key -> PriorityInfo`.

`Scheduler.sync_library()` should call the calculator once per sync and pass the resulting mapping into item processing.

`_process_sync_item()` should update:

- `priority_score`
- `priority_reasons`
- `priority_last_calculated_at`
- `is_priority`

Completed items should clear priority fields during post-processing to avoid stale display and ordering.

## Queue Ordering

Replace automatic ordering by `is_priority DESC` with `priority_score DESC`.

Initial queue ordering:

1. `QUEUED` before `MISSING`
2. `priority_score DESC`
3. `queue_order ASC`
4. `updated_at DESC`

Manual `move_top` and bulk move-top behavior should continue to set `QUEUED` and adjust `queue_order`.

Processing logs should include the top few candidate scores and reason summaries.

## API and UI

The current UI can keep showing the priority badge through `is_priority`.

API responses that already include `is_priority` should also include:

- `priority_score`
- `priority_reasons`

The first UI pass can expose this through tooltip/title text or item details. A full scoring dashboard is not required.

## Error Handling

- If Plex DB priority reads fail, return an empty watch-history signal set and continue sync.
- If hub reads fail or time out for a library, skip hub scoring for that library and continue.
- Cap hub items processed per hub to avoid large sync delays.
- Exclude broad recency hubs to avoid huge candidate sets.

## Testing

Add focused unit tests for the priority calculator:

- Next 3 unwatched episodes are scored; episode 4+ is not.
- On-deck candidates outrank hub candidates.
- Curated hub movies receive a hub score.
- Curated hub episodes receive a hub score.
- Curated hub shows promote only first 3 missing episodes.
- Broad recently-added/released hubs are ignored.
- Multi-user overlap boosts score but respects the cap.
- Old watch history is ignored by recency decay.

Add scheduler tests for:

- Queue ordering uses `priority_score DESC`.
- Manual `QUEUED` items still outrank automatic priority.
- Completed items clear stale priority fields.

## Open Implementation Notes

The first implementation should keep weights hard-coded in the priority module. Settings can be added later if real usage shows the weights need tuning.

The priority calculator should prefer Plex DB queries for watch-derived prediction. PlexAPI should be used only for on-deck and curated hub reads.
