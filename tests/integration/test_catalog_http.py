from uuid import uuid4

import pytest
from dishka import Provider, Scope, provide
from fastapi.testclient import TestClient

from config.settings import Settings
from di.container import create_container
from domain.value_objects.media_type import MediaType
from main import create_app
from port.exceptions.movie_metadata import MovieMetadataNotFoundError, MovieMetadataUnavailableError
from port.movie_metadata import MovieMetadataProvider
from tests.factories import FakeMetadata, metadata


@pytest.fixture
def fake_metadata() -> FakeMetadata:
    return FakeMetadata()


@pytest.fixture
def catalog_client(migrated_settings: Settings, fake_metadata: FakeMetadata):
    class TestMetadataProvider(Provider):
        @provide(scope=Scope.APP, override=True)
        def metadata(self) -> MovieMetadataProvider:
            return fake_metadata

    container = create_container(migrated_settings, TestMetadataProvider())
    with TestClient(create_app(migrated_settings, container=container)) as client:
        yield client


def test_import_read_search_and_reimport(catalog_client: TestClient, fake_metadata: FakeMetadata) -> None:
    assert catalog_client.get("/api/v1/movies").json()["items"] == []
    response = catalog_client.post("/api/v1/catalog/import", json={})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["imported_count"] == 1
    movie = body["items"][0]
    assert movie["tmdb_id"] == 123 and movie["movielens_id"] is None
    assert catalog_client.get(f"/api/v1/movies/{movie['id']}").json() == movie
    # Unicode case folding and original titles work; LIKE wildcards are literal.
    assert catalog_client.get("/api/v1/movies", params={"q": "ТАЙНА"}).json()["total_results"] == 1
    assert catalog_client.get("/api/v1/movies?q=coco").json()["total_results"] == 1
    assert catalog_client.get("/api/v1/movies?q=%25").json()["total_results"] == 0
    assert catalog_client.get("/api/v1/movies?genre_id=16").json()["total_results"] == 1
    assert catalog_client.get("/api/v1/movies?genre_id=99").json()["total_results"] == 0
    fake_metadata.items = (metadata(title="Обновлённое название", genres=()),)
    updated = catalog_client.post("/api/v1/catalog/import", json={}).json()["items"][0]
    assert updated["id"] == movie["id"]
    assert catalog_client.get("/api/v1/movies").json()["total_results"] == 1
    assert catalog_client.get("/api/v1/movies?genre_id=16").json()["total_results"] == 0
    assert catalog_client.get(f"/api/v1/movies/{movie['id']}").json()["genres"] == []


def test_media_identity_pagination_and_popularity(
    catalog_client: TestClient, fake_metadata: FakeMetadata
) -> None:
    fake_metadata.items = (metadata(), metadata(tmdb_id=124, title="Другой фильм", popularity=40))
    movies = catalog_client.post("/api/v1/catalog/import", json={}).json()["items"]
    series = catalog_client.post("/api/v1/catalog/import/tv/123").json()
    assert series["id"] != movies[0]["id"]
    assert catalog_client.get("/api/v1/movies?media_type=tv").json()["total_results"] == 1
    first = catalog_client.get("/api/v1/movies?page_size=1").json()
    assert first["total_pages"] == 2
    assert first["items"][0]["tmdb_id"] == 124
    second = catalog_client.get("/api/v1/movies?page=2&page_size=1").json()
    assert second["items"][0]["tmdb_id"] == 123
    assert catalog_client.get("/api/v1/movies?page=3&page_size=1").json()["items"] == []


def test_remote_outage_leaves_local_catalog_readable(
    catalog_client: TestClient, fake_metadata: FakeMetadata
) -> None:
    catalog_client.post("/api/v1/catalog/import", json={})
    fake_metadata.error = MovieMetadataUnavailableError()
    response = catalog_client.post("/api/v1/catalog/import", json={})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "metadata_unavailable"
    assert catalog_client.get("/api/v1/movies").json()["total_results"] == 1
    fake_metadata.error = MovieMetadataNotFoundError()
    assert catalog_client.post("/api/v1/catalog/import/movie/999").status_code == 404


def test_catalog_import_without_key_is_explicit(migrated_settings: Settings) -> None:
    with TestClient(create_app(migrated_settings)) as client:
        assert client.post("/api/v1/catalog/import", json={}).status_code == 503
        assert client.get("/api/v1/movies").status_code == 200
        assert client.get("/ready").status_code == 200


def test_search_import_arguments_and_duplicate_items(
    catalog_client: TestClient, fake_metadata: FakeMetadata
) -> None:
    fake_metadata.items = (metadata(), metadata())
    result = catalog_client.post("/api/v1/catalog/import", json={"q": " Коко ", "genre_id": 16, "page": 2})
    assert result.status_code == 200
    assert result.json()["imported_count"] == 1
    assert fake_metadata.calls[-1] == {
        "media_type": MediaType.MOVIE,
        "query": "Коко",
        "genre_id": 16,
        "page": 2,
    }


@pytest.mark.parametrize(
    "url",
    [
        "/api/v1/movies?page=0",
        "/api/v1/movies?page_size=101",
        "/api/v1/movies?genre_id=-1",
        "/api/v1/movies?media_type=cartoons",
        "/api/v1/movies/not-a-uuid",
    ],
)
def test_catalog_validates_query(catalog_client: TestClient, url: str) -> None:
    assert catalog_client.get(url).status_code == 422


def test_missing_local_movie_returns_404(catalog_client: TestClient) -> None:
    assert catalog_client.get(f"/api/v1/movies/{uuid4()}").status_code == 404


@pytest.mark.parametrize(
    "body", [{"page": 501}, {"media_type": "cartoons"}, {"genre_id": 0}, {"unknown": True}]
)
def test_import_validates_request(catalog_client: TestClient, body: dict) -> None:
    assert catalog_client.post("/api/v1/catalog/import", json=body).status_code == 422


def test_tmdb_lookup_reads_without_importing(catalog_client: TestClient, fake_metadata: FakeMetadata) -> None:
    assert catalog_client.get("/api/v1/movies/by-tmdb/movie/123").status_code == 404
    assert not fake_metadata.calls
    imported = catalog_client.post("/api/v1/catalog/import", json={}).json()["items"][0]
    fake_metadata.calls.clear()
    assert catalog_client.get("/api/v1/movies/by-tmdb/movie/123").json() == imported
    assert catalog_client.get("/api/v1/movies/by-tmdb/tv/123").status_code == 404
    assert not fake_metadata.calls


def test_discovery_does_not_import_or_assign_uuid(
    catalog_client: TestClient, fake_metadata: FakeMetadata
) -> None:
    response = catalog_client.get("/api/v1/discovery/movies", params={"q": "Новый фильм"})
    assert response.status_code == 200, response.text
    assert response.json()["items"][0]["id"] is None
    assert catalog_client.get("/api/v1/movies").json()["total_results"] == 0
    response = catalog_client.get("/api/v1/discovery/movies/movie/123")
    assert response.status_code == 200
    assert response.json()["tmdb_id"] == 123
    assert catalog_client.get("/api/v1/movies").json()["total_results"] == 0
    # Only an explicit persistence command allocates a local UUID.
    saved = catalog_client.post("/api/v1/catalog/import/movie/123").json()
    assert saved["id"] is not None
    assert catalog_client.get("/api/v1/movies").json()["total_results"] == 1


def test_filtered_discovery_does_not_claim_unfiltered_total(catalog_client: TestClient) -> None:
    response = catalog_client.get("/api/v1/discovery/movies?q=Коко&genre_id=16")
    assert response.status_code == 200
    assert response.json()["total_results"] is None
    assert catalog_client.get("/api/v1/discovery/movies?page=501").status_code == 422


def test_discovery_works_without_opening_database(settings: Settings, fake_metadata: FakeMetadata) -> None:
    from pathlib import Path

    from sqlalchemy.engine import make_url

    class TestMetadataProvider(Provider):
        @provide(scope=Scope.APP, override=True)
        def metadata(self) -> MovieMetadataProvider:
            return fake_metadata

    container = create_container(settings, TestMetadataProvider())
    with TestClient(create_app(settings, container=container)) as client:
        assert client.get("/api/v1/discovery/movies?q=Фильм").status_code == 200
        assert client.get("/api/v1/discovery/movies/movie/123").status_code == 200
    assert not Path(make_url(settings.database_url.get_secret_value()).database).exists()
