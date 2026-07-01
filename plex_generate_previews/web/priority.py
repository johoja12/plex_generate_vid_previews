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


@dataclass(frozen=True)
class EpisodeRow:
    rating_key: int
    show_id: int
    season_index: int
    episode_index: int


@dataclass(frozen=True)
class WatchEvent:
    account_id: int
    rating_key: int
    show_id: int
    season_index: int
    episode_index: int
    viewed_at: datetime


@dataclass(frozen=True)
class HubItem:
    rating_key: int
    item_type: str
    title: str
    show_id: int | None = None
    season_index: int | None = None


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
    return any(
        term in lowered
        for term in ("trending", "popular", "most watched", "favorite", "favourite")
    )


def score_next_up_episodes(
    watch_events: list[WatchEvent],
    episodes: list[EpisodeRow],
    missing_rating_keys: set[int],
    now: datetime,
) -> dict[int, PriorityInfo]:
    latest_by_user_show: dict[tuple[int, int], WatchEvent] = {}
    for event in watch_events:
        if recency_multiplier(event.viewed_at, now) == 0:
            continue
        key = (event.account_id, event.show_id)
        previous = latest_by_user_show.get(key)
        if previous is None or event.viewed_at > previous.viewed_at:
            latest_by_user_show[key] = event

    episodes_by_show: dict[int, list[EpisodeRow]] = {}
    for episode in episodes:
        episodes_by_show.setdefault(episode.show_id, []).append(episode)
    for show_episodes in episodes_by_show.values():
        show_episodes.sort(key=lambda row: (row.season_index, row.episode_index, row.rating_key))

    result: dict[int, PriorityInfo] = {}
    accounts_by_candidate: dict[int, set[int]] = {}

    for event in latest_by_user_show.values():
        candidates = [
            episode
            for episode in episodes_by_show.get(event.show_id, [])
            if episode.rating_key in missing_rating_keys
            and (episode.season_index, episode.episode_index)
            > (event.season_index, event.episode_index)
        ][: len(NEXT_EPISODE_SCORES)]

        multiplier = recency_multiplier(event.viewed_at, now)
        for index, episode in enumerate(candidates):
            info = result.setdefault(episode.rating_key, PriorityInfo())
            accounts = accounts_by_candidate.setdefault(episode.rating_key, set())
            base_score = int(NEXT_EPISODE_SCORES[index] * multiplier)
            if not accounts:
                add_reason(
                    info,
                    "next_episode",
                    base_score,
                    account_id=event.account_id,
                    position=index + 1,
                    source="watch_history",
                )
            accounts.add(event.account_id)

    for rating_key, account_ids in accounts_by_candidate.items():
        additional_users = max(0, len(account_ids) - 1)
        if additional_users:
            boost = min(MULTI_USER_BOOST_CAP, additional_users * MULTI_USER_BOOST)
            add_reason(
                result[rating_key],
                "multi_user_overlap",
                boost,
                user_count=len(account_ids),
            )

    return result


def score_hub_items(
    hub_title: str,
    hub_items: list[HubItem],
    episodes: list[EpisodeRow],
    missing_rating_keys: set[int],
) -> dict[int, PriorityInfo]:
    if not is_included_hub(hub_title):
        return {}

    episodes_by_show: dict[int, list[EpisodeRow]] = {}
    episodes_by_season: dict[tuple[int, int], list[EpisodeRow]] = {}
    for episode in episodes:
        episodes_by_show.setdefault(episode.show_id, []).append(episode)
        episodes_by_season.setdefault((episode.show_id, episode.season_index), []).append(episode)

    for grouped in list(episodes_by_show.values()) + list(episodes_by_season.values()):
        grouped.sort(key=lambda row: (row.season_index, row.episode_index, row.rating_key))

    result: dict[int, PriorityInfo] = {}

    def add_hub_reason(rating_key: int, score: int) -> None:
        info = result.setdefault(rating_key, PriorityInfo())
        add_reason(info, "hub", score, hub=hub_title)

    for item in hub_items:
        if item.item_type in {"movie", "episode"}:
            if item.rating_key in missing_rating_keys:
                add_hub_reason(item.rating_key, HUB_DIRECT_SCORE)
        elif item.item_type == "show":
            for episode in [
                row
                for row in episodes_by_show.get(item.rating_key, [])
                if row.rating_key in missing_rating_keys
            ][:3]:
                add_hub_reason(episode.rating_key, HUB_SHOW_SCORE)
        elif item.item_type == "season" and item.show_id is not None and item.season_index is not None:
            for episode in [
                row
                for row in episodes_by_season.get((item.show_id, item.season_index), [])
                if row.rating_key in missing_rating_keys
            ][:3]:
                add_hub_reason(episode.rating_key, HUB_SHOW_SCORE)

    return result
