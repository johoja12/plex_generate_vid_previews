from datetime import datetime, timedelta, timezone

from plex_generate_previews.web.priority import (
    EpisodeRow,
    HubItem,
    PriorityInfo,
    WatchEvent,
    add_reason,
    is_excluded_hub,
    is_included_hub,
    recency_multiplier,
    score_hub_items,
    score_next_up_episodes,
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


def test_next_three_unwatched_episodes_are_scored_per_user():
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    episodes = [
        EpisodeRow(101, 10, 1, 1),
        EpisodeRow(102, 10, 1, 2),
        EpisodeRow(103, 10, 1, 3),
        EpisodeRow(104, 10, 1, 4),
        EpisodeRow(105, 10, 1, 5),
        EpisodeRow(106, 10, 1, 6),
    ]
    watches = [
        WatchEvent(
            account_id=1,
            rating_key=102,
            show_id=10,
            season_index=1,
            episode_index=2,
            viewed_at=now,
        )
    ]

    result = score_next_up_episodes(
        watches, episodes, missing_rating_keys={103, 104, 105, 106}, now=now
    )

    assert [result[key].score for key in (103, 104, 105)] == [800, 760, 720]
    assert 101 not in result
    assert 102 not in result
    assert 106 not in result


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
        WatchEvent(
            account_id=1,
            rating_key=201,
            show_id=20,
            season_index=1,
            episode_index=1,
            viewed_at=now,
        ),
        WatchEvent(
            account_id=2,
            rating_key=301,
            show_id=30,
            season_index=1,
            episode_index=1,
            viewed_at=now,
        ),
    ]

    result = score_next_up_episodes(
        watches, episodes, missing_rating_keys={202, 203, 302}, now=now
    )

    assert 202 in result
    assert 203 in result
    assert 302 in result


def test_shared_next_up_candidate_gets_overlap_boost():
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    episodes = [EpisodeRow(401, 40, 1, 1), EpisodeRow(402, 40, 1, 2)]
    watches = [
        WatchEvent(
            account_id=1,
            rating_key=401,
            show_id=40,
            season_index=1,
            episode_index=1,
            viewed_at=now,
        ),
        WatchEvent(
            account_id=2,
            rating_key=401,
            show_id=40,
            season_index=1,
            episode_index=1,
            viewed_at=now,
        ),
    ]

    result = score_next_up_episodes(watches, episodes, missing_rating_keys={402}, now=now)

    assert result[402].score == 875
    assert [reason["type"] for reason in result[402].reasons] == [
        "next_episode",
        "multi_user_overlap",
    ]


def test_old_watch_history_does_not_create_next_up_candidate():
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    episodes = [EpisodeRow(501, 50, 1, 1), EpisodeRow(502, 50, 1, 2)]
    watches = [
        WatchEvent(
            account_id=1,
            rating_key=501,
            show_id=50,
            season_index=1,
            episode_index=1,
            viewed_at=now - timedelta(days=31),
        )
    ]

    result = score_next_up_episodes(watches, episodes, missing_rating_keys={502}, now=now)

    assert result == {}


def test_hub_direct_movie_or_episode_gets_direct_score():
    items = [
        HubItem(rating_key=601, item_type="movie", title="Movie"),
        HubItem(rating_key=602, item_type="episode", title="Episode"),
    ]

    result = score_hub_items(
        "Trending",
        items,
        episodes=[],
        missing_rating_keys={601, 602},
    )

    assert result[601].score == 300
    assert result[602].score == 300
    assert result[601].reasons[0]["hub"] == "Trending"


def test_hub_show_promotes_first_three_missing_episodes():
    items = [HubItem(rating_key=70, item_type="show", title="Show")]
    episodes = [
        EpisodeRow(701, 70, 1, 1),
        EpisodeRow(702, 70, 1, 2),
        EpisodeRow(703, 70, 1, 3),
        EpisodeRow(704, 70, 1, 4),
    ]

    result = score_hub_items(
        "Popular TV This Year",
        items,
        episodes=episodes,
        missing_rating_keys={701, 702, 703, 704},
    )

    assert set(result) == {701, 702, 703}
    assert all(info.score == 250 for info in result.values())


def test_hub_season_promotes_first_three_missing_episodes_in_that_season():
    items = [HubItem(rating_key=8001, item_type="season", title="Season", show_id=80, season_index=2)]
    episodes = [
        EpisodeRow(801, 80, 1, 1),
        EpisodeRow(802, 80, 2, 1),
        EpisodeRow(803, 80, 2, 2),
        EpisodeRow(804, 80, 2, 3),
        EpisodeRow(805, 80, 2, 4),
    ]

    result = score_hub_items(
        "Most Watched This Week",
        items,
        episodes=episodes,
        missing_rating_keys={801, 802, 803, 804, 805},
    )

    assert set(result) == {802, 803, 804}


def test_recency_hub_items_are_ignored():
    result = score_hub_items(
        "Recently Added Popular Movies",
        [HubItem(rating_key=901, item_type="movie", title="Movie")],
        episodes=[],
        missing_rating_keys={901},
    )

    assert result == {}
