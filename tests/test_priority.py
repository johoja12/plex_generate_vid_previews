from datetime import datetime, timedelta, timezone

from plex_generate_previews.web.priority import (
    PriorityInfo,
    add_reason,
    is_excluded_hub,
    is_included_hub,
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
