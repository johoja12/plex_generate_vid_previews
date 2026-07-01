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
    return any(
        term in lowered
        for term in ("trending", "popular", "most watched", "favorite", "favourite")
    )
