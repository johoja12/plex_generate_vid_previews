import os
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from plex_generate_previews.web.main import app, get_session, get_current_user
from plex_generate_previews.web.models import MediaItem, PreviewStatus, MediaType

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
