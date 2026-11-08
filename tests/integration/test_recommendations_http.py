from uuid import uuid4

import pytest
from dishka import Provider, Scope, provide
from fastapi.testclient import TestClient

from config.settings import Settings
from di.container import create_container
from main import create_app
from port.exceptions.movie_metadata import MovieMetadataUnavailableError
from port.movie_metadata import MovieMetadataProvider
from port.recommendation import RankedMovie, RankedRecommendations, RecommendationEngine
from tests.factories import FakeMetadata


class FakeEngine:
    def __init__(self):
        self.calls = []

    async def status(self):
        return "ready"

    async def explore(self, ratings, limit, seed):
        self.calls.append((ratings, limit, seed))
        return RankedRecommendations(
            tuple(
                RankedMovie(identity, str(identity), False, 0.5, exploratory=identity > 910)
                for identity in range(900, 940)
            ),
            "item_knn",
            "fixture-v1",
            len(ratings),
            None,
            mode="surprise",
            seed=seed,
        )

    async def recommend(self, ratings, limit):
        self.calls.append((ratings, limit))
        return RankedRecommendations(
            (RankedMovie(900, "Offline title", False, 0.7),),
            "item_knn" if ratings else "popularity",
            "fixture-v1",
            len(ratings),
            None if ratings else "no_known_positive_ratings",
        )


@pytest.fixture
def recommendations_client(migrated_settings):
    engine, metadata = FakeEngine(), FakeMetadata()

    class Overrides(Provider):
        @provide(scope=Scope.APP, override=True)
        def recommender(self) -> RecommendationEngine:
            return engine

        @provide(scope=Scope.APP, override=True)
        def source(self) -> MovieMetadataProvider:
            return metadata

    container = create_container(migrated_settings, Overrides())
    with TestClient(create_app(migrated_settings, container=container)) as client:
        yield client, engine, metadata


def profile(client, name):
    return client.post("/api/v1/profiles", json={"name": name}).json()["id"]


def test_recommendations_use_only_selected_profile_movie_ratings_and_never_import(recommendations_client):
    client, engine, metadata = recommendations_client
    a, b = profile(client, "A"), profile(client, "B")
    client.put(f"/api/v1/profiles/{a}/ratings", json={"media_type": "movie", "tmdb_id": 123, "value": 5})
    client.put(f"/api/v1/profiles/{a}/ratings", json={"media_type": "tv", "tmdb_id": 124, "value": 4})
    before = client.get("/api/v1/movies").json()["total_results"]
    response = client.get(f"/api/v1/profiles/{a}/recommendations?limit=5")
    assert response.status_code == 200, response.text
    body = response.json()
    assert engine.calls[-1] == ({123: 5}, 40)
    assert body["strategy"] == "item_knn" and body["movie_ratings"] == 1
    assert body["model_version"] == "fixture-v1" and body["known_ratings"] == 1
    assert body["items"][0]["movie"]["id"] is None
    assert body["items"][0]["movie"]["tmdb_id"] == 900
    assert body["metadata_partial"] is False
    other = client.get(f"/api/v1/profiles/{b}/recommendations").json()
    assert engine.calls[-1] == ({}, 40)
    assert other["strategy"] == "popularity" and other["fallback_reason"] == "no_known_positive_ratings"
    assert client.get("/api/v1/movies").json()["total_results"] == before
    metadata.error = MovieMetadataUnavailableError()
    fallback = client.get(f"/api/v1/profiles/{a}/recommendations").json()
    assert fallback["metadata_partial"] is True
    assert fallback["items"][0]["movie"]["title"] == "Offline title"
    assert fallback["items"][0]["movie"]["poster_path"] is None


def test_unknown_profile_and_invalid_limit_do_not_call_model(recommendations_client):
    client, engine, metadata = recommendations_client
    assert client.get(f"/api/v1/profiles/{uuid4()}/recommendations").status_code == 404
    for value in (0, 51, "invalid"):
        assert client.get(f"/api/v1/profiles/{uuid4()}/recommendations?limit={value}").status_code == 422
    assert not engine.calls and not metadata.calls


def test_unconfigured_and_missing_model_are_explicit(migrated_settings: Settings, tmp_path):
    with TestClient(create_app(migrated_settings)) as client:
        id_ = profile(client, "A")
        result = client.get(f"/api/v1/profiles/{id_}/recommendations")
        assert result.status_code == 503
        assert result.json()["error"]["code"] == "recommendation_unavailable"
        assert client.get("/ready").json()["model"] == "not_configured"
    missing = migrated_settings.model_copy(update={"recommendation_model_path": tmp_path / "missing"})
    with TestClient(create_app(missing)) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200 and ready.json()["model"] == "unavailable"
        assert client.get(f"/api/v1/profiles/{id_}/library").status_code == 200


def test_service_works_without_model_and_returns_fresh_content(migrated_settings):
    from port.movie_metadata import MetadataPage
    from tests.factories import metadata

    class FreshSource(FakeMetadata):
        async def get_page(self, **kwargs):
            self.calls.append(kwargs)
            return MetadataPage((metadata(tmdb_id=1000001, title="Новый фильм"),), 1, 1, 1)

    source = FreshSource()

    class Override(Provider):
        @provide(scope=Scope.APP, override=True)
        def metadata(self) -> MovieMetadataProvider:
            return source

    container = create_container(migrated_settings, Override())
    with TestClient(create_app(migrated_settings, container=container)) as client:
        id_ = profile(client, "Без ML")
        base = client.get(f"/api/v1/profiles/{id_}/recommendations").json()
        assert base["strategy"] == "popularity" and base["model_version"] is None
        client.put(
            f"/api/v1/profiles/{id_}/ratings", json={"media_type": "movie", "tmdb_id": 123, "value": 5}
        )
        result = client.get(f"/api/v1/profiles/{id_}/recommendations").json()
        assert result["strategy"] == "content" and result["model_version"] is None
        assert result["items"][0]["movie"]["title"] == "Новый фильм"
        assert result["items"][0]["sources"] == ["content"]
        assert result["known_ratings"] == 0 and result["movie_ratings"] == 1
        assert client.get("/api/v1/movies").json()["total_results"] == 1
        assert client.get("/ready").json()["model"] == "not_configured"


def test_surprise_http_validates_query_is_profile_scoped_and_never_imports(recommendations_client):
    client, engine, metadata = recommendations_client
    a, b = profile(client, "A"), profile(client, "B")
    assert (
        client.put(
            f"/api/v1/profiles/{a}/ratings", json={"media_type": "movie", "tmdb_id": 900, "value": 5}
        ).status_code
        == 200
    )
    before = client.get("/api/v1/movies").json()["total_results"]
    url = f"/api/v1/profiles/{a}/recommendations?mode=surprise&seed=0"
    response = client.get(url)
    assert response.status_code == 200, response.text
    first = response.json()
    assert first["mode"] == "surprise" and first["seed"] == 0
    assert engine.calls[-1] == ({900: 5}, 40, 0)
    assert len(first["items"]) == 10
    assert 900 not in {item["movie"]["tmdb_id"] for item in first["items"]}
    assert first == client.get(url).json()
    assert client.get(f"/api/v1/profiles/{b}/recommendations?mode=surprise&seed=1").status_code == 200
    assert engine.calls[-1] == ({}, 40, 1)
    fresh = client.get(f"/api/v1/profiles/{a}/recommendations?mode=surprise").json()
    assert 0 <= fresh["seed"] < 2**32
    assert client.get("/api/v1/movies").json()["total_results"] == before
    assert client.get(f"/api/v1/profiles/{a}/recommendations").json()["mode"] == "personalized"
    for query in ("mode=invalid", "mode=surprise&seed=-1", "seed=4294967296", "seed=bad"):
        assert client.get(f"/api/v1/profiles/{a}/recommendations?{query}").status_code == 422


def test_larger_personalized_limit_is_accepted_but_surprise_stays_bounded(recommendations_client):
    client, engine, _ = recommendations_client
    id_ = profile(client, "Larger selection")
    response = client.get(f"/api/v1/profiles/{id_}/recommendations?limit=50")
    assert response.status_code == 200, response.text
    assert engine.calls[-1] == ({}, 100)
    assert client.get(f"/api/v1/profiles/{id_}/recommendations?mode=surprise&limit=21").status_code == 422
