import os
import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from plex_generate_previews.web.main import app, get_session, get_current_user
from plex_generate_previews.web.models import MediaItem, PreviewStatus, MediaType
from pathlib import Path

# Setup in-memory DB
@pytest.fixture(name="session")
def session_fixture():
    engine = create_engine(
        "sqlite://", 
        connect_args={"check_same_thread": False}, 
        poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

@pytest.fixture(name="client")
def client_fixture(session: Session):
    def get_session_override():
        return session

    def get_current_user_override():
        return "testuser"

    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_current_user] = get_current_user_override
    
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()

def test_get_items_with_symlink(client, session, tmp_path):
    # Create a dummy file
    target_file = tmp_path / "target_video.mkv"
    target_file.touch()
    
    # Create a symlink
    symlink_file = tmp_path / "symlink_video.mkv"
    symlink_file.symlink_to(target_file)
    
    # Create MediaItem
    item = MediaItem(
        id=1,
        title="Symlink Movie",
        library_name="Movies",
        media_type=MediaType.MOVIE,
        file_path=str(symlink_file),
        status=PreviewStatus.MISSING
    )
    session.add(item)
    session.commit()
    
    response = client.get("/api/items")
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["items"]) == 1
    item_data = data["items"][0]
    assert item_data["file_path"] == str(symlink_file)
    assert item_data["symlink_target"] == str(target_file)

def test_get_items_without_symlink(client, session, tmp_path):
    # Create a regular file
    regular_file = tmp_path / "regular_video.mkv"
    regular_file.touch()
    
    item = MediaItem(
        id=2,
        title="Regular Movie",
        library_name="Movies",
        media_type=MediaType.MOVIE,
        file_path=str(regular_file),
        status=PreviewStatus.MISSING
    )
    session.add(item)
    session.commit()
    
    response = client.get("/api/items")
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["items"]) == 1
    item_data = data["items"][0]
    assert item_data["file_path"] == str(regular_file)
    assert item_data["symlink_target"] is None


def test_stats_include_priority_count(client, session):
    priority_item = MediaItem(
        id=3,
        title="Priority Movie",
        library_name="Movies",
        media_type=MediaType.MOVIE,
        status=PreviewStatus.MISSING,
        is_priority=True,
        priority_score=300,
    )
    regular_item = MediaItem(
        id=4,
        title="Regular Movie",
        library_name="Movies",
        media_type=MediaType.MOVIE,
        status=PreviewStatus.MISSING,
    )
    session.add(priority_item)
    session.add(regular_item)
    session.commit()

    response = client.get("/api/stats")

    assert response.status_code == 200
    assert response.json()["priority"] == 1


def test_dashboard_renders_priority_stat_card():
    html = Path("plex_generate_previews/web/templates/index.html").read_text()

    assert "Priority" in html
    assert 'x-text="stats.priority"' in html


def test_get_items_priority_pagination_uses_total_matching_rows(client, session):
    for index in range(120):
        session.add(
            MediaItem(
                id=1000 + index,
                title=f"Priority Movie {index}",
                library_name="Movies",
                media_type=MediaType.MOVIE,
                status=PreviewStatus.MISSING,
                is_priority=True,
                priority_score=300,
            )
        )
    session.commit()

    response = client.get("/api/items?show_priority_only=true&page=1&limit=50")

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 50
    assert data["total"] == 120
    assert data["pages"] == 3


def test_get_items_default_order_matches_processing_priority(client, session):
    newest = datetime(2026, 7, 1, 12, 0, 0)
    older = datetime(2026, 6, 1, 12, 0, 0)

    session.add(
        MediaItem(
            id=2001,
            title="Newest Non Priority",
            library_name="Movies",
            media_type=MediaType.MOVIE,
            status=PreviewStatus.MISSING,
            priority_score=0,
            added_at=newest,
            updated_at=newest,
        )
    )
    session.add(
        MediaItem(
            id=2002,
            title="Low Priority",
            library_name="Movies",
            media_type=MediaType.MOVIE,
            status=PreviewStatus.MISSING,
            is_priority=True,
            priority_score=250,
            added_at=older,
            updated_at=older,
        )
    )
    session.add(
        MediaItem(
            id=2003,
            title="High Priority",
            library_name="Movies",
            media_type=MediaType.MOVIE,
            status=PreviewStatus.MISSING,
            is_priority=True,
            priority_score=900,
            added_at=older,
            updated_at=older,
        )
    )
    session.commit()

    response = client.get("/api/items?hide_completed=true&page=1&limit=50")

    assert response.status_code == 200
    titles = [item["title"] for item in response.json()["items"]]
    assert titles[:3] == ["High Priority", "Low Priority", "Newest Non Priority"]


def test_get_items_default_order_keeps_processable_items_before_failed(client, session):
    now = datetime(2026, 7, 1, 12, 0, 0)

    session.add(
        MediaItem(
            id=2101,
            title="Failed High Priority",
            library_name="TV",
            media_type=MediaType.EPISODE,
            status=PreviewStatus.FAILED,
            is_priority=True,
            priority_score=999,
            added_at=now,
            updated_at=now,
        )
    )
    session.add(
        MediaItem(
            id=2102,
            title="Missing Lower Priority",
            library_name="TV",
            media_type=MediaType.EPISODE,
            status=PreviewStatus.MISSING,
            is_priority=True,
            priority_score=500,
            added_at=now,
            updated_at=now,
        )
    )
    session.commit()

    response = client.get("/api/items?hide_completed=true&page=1&limit=50")

    assert response.status_code == 200
    titles = [item["title"] for item in response.json()["items"]]
    assert titles[:2] == ["Missing Lower Priority", "Failed High Priority"]
