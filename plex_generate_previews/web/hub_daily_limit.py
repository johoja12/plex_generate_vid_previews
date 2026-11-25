"""Persist the rolling TV hub allowance across priority recalculations."""

from datetime import timedelta

from sqlalchemy import text


def limit_hub_episodes(engine, infos, episodes, hub_items, now):
    """Reserve at most three distinct hub episodes per show in 24 hours.

    Reservations are shared by all hubs and survive completion and restarts.
    Other priority reasons are unaffected. A missing show mapping fails closed.
    """
    shows = {episode.rating_key: episode.show_id for episode in episodes}
    direct_episodes = {
        item.rating_key for items in hub_items.values() for item in items
        if item.item_type == "episode"
    }
    candidates = [
        (key, info) for key, info in infos.items()
        if any(reason.get("type") == "hub" for reason in info.reasons)
        and (key in shows or key in direct_episodes)
    ]
    if not candidates:
        return

    with engine.connect() as connection:
        try:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            connection.exec_driver_sql(
                "CREATE TABLE IF NOT EXISTS hub_episode_reservation ("
                "show_id INTEGER NOT NULL, episode_id INTEGER NOT NULL, "
                "reserved_at TEXT NOT NULL, PRIMARY KEY (show_id, episode_id))"
            )
            cutoff = (now - timedelta(hours=24)).isoformat()
            connection.execute(text("DELETE FROM hub_episode_reservation WHERE reserved_at <= :cutoff"), {"cutoff": cutoff})
            rows = connection.execute(text("SELECT show_id, episode_id FROM hub_episode_reservation")).all()
            reserved = {}
            for show_id, episode_id in rows:
                reserved.setdefault(show_id, set()).add(episode_id)
            for key, info in candidates:
                show_id = shows.get(key)
                selected = reserved.setdefault(show_id, set())
                allowed = show_id is not None and (key in selected or len(selected) < 3)
                if allowed and key not in selected:
                    connection.execute(text("INSERT INTO hub_episode_reservation VALUES (:show, :episode, :at)"),
                                       {"show": show_id, "episode": key, "at": now.isoformat()})
                    selected.add(key)
                if not allowed:
                    info.reasons = [reason for reason in info.reasons if reason.get("type") != "hub"]
                    info.score = min(999, sum(int(reason.get("score", 0)) for reason in info.reasons))
                    if not info.reasons:
                        del infos[key]
            connection.commit()
        except Exception:
            connection.rollback()
            raise
