from uuid import uuid4

import pytest
from dishka import Provider, Scope, provide
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from adapter.database.repositories.library import SQLAlchemyLibraryRepository
from config.settings import Settings
from di.container import create_container
from main import create_app
from port.exceptions.movie_metadata import MovieMetadataUnavailableError
from port.movie_metadata import MovieMetadataProvider
from port.repositories.library import LibraryRepository
from tests.factories import FakeMetadata


def make_client(settings: Settings, metadata: FakeMetadata, *extra: Provider) -> TestClient:
    class TestMetadataProvider(Provider):
        @provide(scope=Scope.APP, override=True)
        def source(self) -> MovieMetadataProvider:
            return metadata

    return TestClient(
        create_app(settings, container=create_container(settings, TestMetadataProvider(), *extra))
    )


@pytest.fixture
def metadata() -> FakeMetadata:
    return FakeMetadata()


@pytest.fixture
def client(migrated_settings: Settings, metadata: FakeMetadata):
    with make_client(migrated_settings, metadata) as client:
        yield client


def profile(client: TestClient, name="User") -> str:
    return client.post("/api/v1/profiles", json={"name": name}).json()["id"]


def ref(tmdb_id=123, **extra) -> dict:
    return {"media_type": "movie", "tmdb_id": tmdb_id, **extra}


def test_profiles_have_independent_ratings_and_favorites(client: TestClient) -> None:
    a, b = profile(client, "A"), profile(client, "B")
    first = client.put(f"/api/v1/profiles/{a}/favorites", json=ref())
    assert first.status_code == 200, first.text
    movie_id = first.json()["movie"]["id"]
    again = client.put(f"/api/v1/profiles/{a}/favorites/{movie_id}")
    assert again.json()["created_at"] == first.json()["created_at"]
    assert client.put(f"/api/v1/profiles/{a}/ratings", json=ref(value=2)).status_code == 200
    assert client.put(f"/api/v1/profiles/{b}/ratings/{movie_id}", json={"value": 5}).status_code == 200
    library_a = client.get(f"/api/v1/profiles/{a}/library").json()
    library_b = client.get(f"/api/v1/profiles/{b}/library").json()
    assert len(library_a["favorites"]) == 1 and library_a["ratings"][0]["value"] == 2
    assert library_b["favorites"] == [] and library_b["ratings"][0]["value"] == 5
    assert client.delete(f"/api/v1/profiles/{a}/ratings/{movie_id}").status_code == 204
    assert client.delete(f"/api/v1/profiles/{a}/ratings/{movie_id}").status_code == 204
    assert client.get(f"/api/v1/profiles/{a}/library").json()["ratings"] == []
    assert client.get(f"/api/v1/profiles/{b}/library").json()["ratings"][0]["value"] == 5
    assert client.delete(f"/api/v1/profiles/{a}/favorites/{movie_id}").status_code == 204
    assert client.get(f"/api/v1/profiles/{a}/library").json()["favorites"] == []
    assert client.get("/api/v1/movies").json()["total_results"] == 1


def test_existing_card_actions_work_without_tmdb(client: TestClient, metadata: FakeMetadata) -> None:
    user = profile(client)
    first = client.put(f"/api/v1/profiles/{user}/ratings", json=ref(value=2)).json()
    metadata.error = MovieMetadataUnavailableError()
    metadata.calls.clear()
    updated = client.put(f"/api/v1/profiles/{user}/ratings", json=ref(value=4))
    assert updated.status_code == 200 and updated.json()["value"] == 4
    assert updated.json()["movie"]["id"] == first["movie"]["id"]
    assert metadata.calls == []
    assert client.put(f"/api/v1/profiles/{user}/favorites", json=ref(tmdb_id=999)).status_code == 502
    assert client.get(f"/api/v1/profiles/{user}/library").json()["favorites"] == []


def test_unknown_profile_does_not_import_movies(client: TestClient, metadata: FakeMetadata) -> None:
    missing = uuid4()
    assert client.put(f"/api/v1/profiles/{missing}/favorites", json=ref()).status_code == 404
    assert client.get(f"/api/v1/profiles/{missing}/library").status_code == 404
    assert metadata.calls == []
    user = profile(client)
    assert client.put(f"/api/v1/profiles/{user}/ratings/{uuid4()}", json={"value": 4}).status_code == 404
    assert client.get("/api/v1/movies").json()["total_results"] == 0


@pytest.mark.parametrize("value", [0, 6, 2.5, True, "4", None])
def test_invalid_rating_never_imports_or_persists(client: TestClient, metadata: FakeMetadata, value) -> None:
    user = profile(client)
    assert client.put(f"/api/v1/profiles/{user}/ratings", json=ref(value=value)).status_code == 422
    assert client.get(f"/api/v1/profiles/{user}/library").json()["ratings"] == []
    assert metadata.calls == []


def test_library_survives_application_restart(migrated_settings: Settings, metadata: FakeMetadata) -> None:
    with make_client(migrated_settings, metadata) as client:
        user = profile(client)
        client.put(f"/api/v1/profiles/{user}/ratings", json=ref(value=4))
        client.put(f"/api/v1/profiles/{user}/favorites", json=ref())
        original = client.get(f"/api/v1/profiles/{user}/library").json()
    with make_client(migrated_settings, metadata) as client:
        assert client.get(f"/api/v1/profiles/{user}/library").json() == original


def test_legacy_import_preserves_server_rating_and_is_repeatable(client: TestClient) -> None:
    user = profile(client)
    server = client.put(f"/api/v1/profiles/{user}/ratings", json=ref(value=5)).json()
    entries = [
        ref(favorite=True, rating=2, rated_at="2020-01-01T00:00:00Z"),
        ref(tmdb_id=124, favorite=True, rating=3, rated_at="2020-01-01T00:00:00Z"),
    ]
    for _ in range(2):
        response = client.post(f"/api/v1/profiles/{user}/library/import", json={"entries": entries})
        assert response.status_code == 200, response.text
    snapshot = client.get(f"/api/v1/profiles/{user}/library").json()
    assert len(snapshot["favorites"]) == 2 and len(snapshot["ratings"]) == 2
    ratings = {row["movie"]["tmdb_id"]: row for row in snapshot["ratings"]}
    assert ratings[123]["value"] == 5 and ratings[123]["updated_at"] == server["updated_at"]
    assert ratings[124]["value"] == 3 and ratings[124]["updated_at"] == "2020-01-01T00:00:00Z"


def test_legacy_duplicate_entries_merge_actions(client: TestClient) -> None:
    user = profile(client)
    result = client.post(
        f"/api/v1/profiles/{user}/library/import", json={"entries": [ref(favorite=True), ref(rating=4)]}
    )
    assert result.status_code == 200
    data = client.get(f"/api/v1/profiles/{user}/library").json()
    assert len(data["favorites"]) == 1 and data["ratings"][0]["value"] == 4


def test_legacy_import_rejects_large_and_naive_timestamp_batches(client: TestClient) -> None:
    user = profile(client)
    for entries in ([ref(favorite=True)] * 21, [ref(rating=3, rated_at="2020-01-01T00:00:00")]):
        assert (
            client.post(f"/api/v1/profiles/{user}/library/import", json={"entries": entries}).status_code
            == 422
        )


def test_movie_and_action_rollback_together(migrated_settings: Settings, metadata: FakeMetadata) -> None:
    class FailingLibrary(SQLAlchemyLibraryRepository):
        async def put_rating(self, rating, *, overwrite=True):
            await super().put_rating(rating, overwrite=overwrite)
            raise RuntimeError("Failure after rating insert")

    class FailureProvider(Provider):
        @provide(scope=Scope.REQUEST, override=True)
        def library(self, session: AsyncSession) -> LibraryRepository:
            return FailingLibrary(session)

    with make_client(migrated_settings, metadata, FailureProvider()) as client:
        user = profile(client)
        with pytest.raises(RuntimeError, match="Failure after rating insert"):
            client.put(f"/api/v1/profiles/{user}/ratings", json=ref(value=4))
        assert client.get("/api/v1/movies").json()["total_results"] == 0
        assert client.get(f"/api/v1/profiles/{user}/library").json() == {"favorites": [], "ratings": []}
        with pytest.raises(RuntimeError):
            client.post(
                f"/api/v1/profiles/{user}/library/import",
                json={"entries": [ref(favorite=True), ref(tmdb_id=124, rating=4)]},
            )
        assert client.get("/api/v1/movies").json()["total_results"] == 0
        assert client.get(f"/api/v1/profiles/{user}/library").json() == {"favorites": [], "ratings": []}
