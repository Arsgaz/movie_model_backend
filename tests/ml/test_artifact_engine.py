import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from adapter.recommendation.item_engine import ArtifactRecommendationEngine  # noqa: E402
from ml.dataset import Film, Observation  # noqa: E402
from ml.model import ItemModel, Parameters  # noqa: E402


@pytest.mark.asyncio
async def test_real_artifact_adapter_is_ready_and_personalizes_new_profile(tmp_path):
    films = [Film(1, 100, "A", "Drama"), Film(2, 200, "B", "Animation")]
    rows = [Observation(user, movie, 5, 1) for user in (1, 2) for movie in (1, 2)]
    model = ItemModel.fit(films, rows, Parameters())
    model.save(tmp_path / "model", {"strategy": "item_knn", "model_version": "test"})
    engine = ArtifactRecommendationEngine(tmp_path / "model")
    assert await engine.status() == "ready"
    ranked = await engine.recommend({100: 5}, 10)
    assert ranked.strategy == "item_knn" and ranked.known_ratings == 1
    assert len(ranked.items) == 1 and ranked.items[0].tmdb_id == 200 and ranked.items[0].animation
    cold = await engine.recommend({999: 5}, 10)
    assert cold.strategy == "popularity" and cold.fallback_reason == "no_known_positive_ratings"


def test_http_loads_real_artifact_and_applies_profile_history(migrated_settings, tmp_path):
    from dishka import Provider, Scope, provide
    from fastapi.testclient import TestClient

    from di.container import create_container
    from main import create_app
    from port.exceptions.movie_metadata import MovieMetadataUnavailableError
    from port.movie_metadata import MovieMetadataProvider
    from tests.factories import FakeMetadata

    films = [Film(1, 100, "A", "Drama"), Film(2, 200, "B", "Animation")]
    rows = [Observation(user, movie, 5, 1) for user in (1, 2) for movie in (1, 2)]
    model = ItemModel.fit(films, rows, Parameters())
    model.save(tmp_path / "model", {"strategy": "item_knn", "model_version": "test"})
    settings = migrated_settings.model_copy(update={"recommendation_model_path": tmp_path / "model"})
    source = FakeMetadata()

    class Source(Provider):
        @provide(scope=Scope.APP, override=True)
        def metadata(self) -> MovieMetadataProvider:
            return source

    with TestClient(create_app(settings, container=create_container(settings, Source()))) as client:
        profile = client.post("/api/v1/profiles", json={"name": "Real model test"}).json()["id"]
        client.put(
            f"/api/v1/profiles/{profile}/ratings", json={"media_type": "movie", "tmdb_id": 100, "value": 5}
        )
        source.error = MovieMetadataUnavailableError()
        result = client.get(f"/api/v1/profiles/{profile}/recommendations").json()
        assert result["strategy"] == "item_knn" and result["known_ratings"] == 1
        assert [item["movie"]["tmdb_id"] for item in result["items"]] == [200]
        assert result["metadata_partial"] is True
        assert result["items"][0]["movie"]["genres"][0]["tmdb_id"] == 16
        assert client.get("/api/v1/movies").json()["total_results"] == 1
        assert client.get("/ready").json()["model"] == "ready"


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_poison_shared_model_loading(monkeypatch, tmp_path):
    import asyncio
    import threading

    started, released = threading.Event(), threading.Event()
    engine = ArtifactRecommendationEngine(tmp_path)

    def load():
        started.set()
        assert released.wait(timeout=2)

    monkeypatch.setattr(engine, "_load", load)
    waiting = asyncio.create_task(engine.status())
    assert await asyncio.to_thread(started.wait, 1)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    released.set()
    assert await engine.status() == "ready"
    await engine.close()


@pytest.mark.asyncio
async def test_artifact_size_limit_prevents_loading_large_weights(tmp_path):
    for name in ("manifest.json", "movies.json", "similarities.npz", "popularity.npy"):
        (tmp_path / name).write_bytes(b"0123456789")
    engine = ArtifactRecommendationEngine(tmp_path, max_bytes=1)
    assert await engine.status() == "unavailable"
    await engine.close()


@pytest.mark.asyncio
async def test_inference_cache_shares_equal_profiles_and_invalidates_changed_ratings(tmp_path, monkeypatch):
    import asyncio

    model = ItemModel.fit(
        [Film(1, 100, "A", "Drama"), Film(2, 200, "B", "Drama")],
        [Observation(user, movie, 5, 1) for user in (1, 2) for movie in (1, 2)],
        Parameters(),
    )
    model.save(tmp_path / "model", {"strategy": "item_knn", "model_version": "cache-test"})
    engine = ArtifactRecommendationEngine(tmp_path / "model")
    calls = []
    original = engine._recommend

    def tracked(ratings, limit):
        calls.append((ratings, limit))
        return original(ratings, limit)

    monkeypatch.setattr(engine, "_recommend", tracked)
    results = await asyncio.gather(*(engine.recommend({100: 5}, 10) for _ in range(8)))
    assert all(result == results[0] for result in results)
    assert len(calls) == 1
    await engine.recommend({100: 4}, 10)
    assert len(calls) == 2
    await engine.close()
