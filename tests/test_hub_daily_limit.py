from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine

from plex_generate_previews.web.hub_daily_limit import limit_hub_episodes
from plex_generate_previews.web.priority import EpisodeRow, HubItem, PriorityInfo


def hub(score=250):
    return PriorityInfo(score=score, reasons=[{"type": "hub", "score": score, "hub": "Trending"}])


def test_limit_persists_across_refresh_hubs_and_24_hours(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    now = datetime(2026, 11, 26, tzinfo=timezone.utc)
    episodes = [EpisodeRow(i, 1, 1, i) for i in range(1, 7)]
    items = {"Trending": [HubItem(1, "show", "Show")],
             "Favourite TV": [HubItem(2, "season", "Season", 1, 1)]}
    first = {i: hub() for i in (1, 2, 3, 4)}
    limit_hub_episodes(engine, first, episodes, items, now)
    assert set(first) == {1, 2, 3}

    later = {i: hub() for i in (2, 4, 5)}
    later[5].reasons.append({"type": "on_deck", "score": 900})
    limit_hub_episodes(engine, later, episodes, items, now + timedelta(hours=1))
    assert set(later) == {2, 5}
    assert later[5].reasons == [{"type": "on_deck", "score": 900}]
    assert later[5].score == 900

    tomorrow = {i: hub() for i in (4, 5, 6)}
    limit_hub_episodes(engine, tomorrow, episodes, items, now + timedelta(hours=24))
    assert set(tomorrow) == {4, 5, 6}


def test_direct_episode_without_show_fails_closed_movie_unaffected(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    items = {"Trending": [HubItem(10, "episode", "Episode"), HubItem(11, "movie", "Movie")]}
    infos = {10: hub(300), 11: hub(300)}
    limit_hub_episodes(engine, infos, [], items, datetime.now(timezone.utc))
    assert set(infos) == {11}
