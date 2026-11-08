import numpy as np
from scipy import sparse

from adapter.recommendation.item_engine import ArtifactRecommendationEngine
from ml.dataset import Film
from ml.model import ItemModel, Parameters


def model():
    n = 1000
    matrix = sparse.lil_matrix((n, n))
    matrix[0, 2:32] = 0.5
    matrix[1, 32:62] = 0.5
    return ItemModel(
        [Film(i, i + 100, str(i), "Drama") for i in range(n)],
        matrix.tocsr(),
        np.arange(n, 0, -1, dtype=float),
        Parameters(),
    )


def test_exploration_samples_beyond_top_lists_and_respects_negative_ratings():
    engine = model()
    ratings = {100: 5, 101: 1}
    first = engine.explore(ratings, limit=10, seed=42)
    assert first == engine.explore(ratings, limit=10, seed=42)
    assert first != engine.explore(ratings, limit=10, seed=43)
    ids = {item.tmdb_id for item, _ in first}
    assert len(ids) == 10
    assert not ids & {100, 101, *range(132, 162)}
    assert sum(broad for _, broad in first) == 3
    assert any(identity > 200 for identity in ids)
    cold = engine.explore({}, limit=10, seed=42)
    assert all(broad for _, broad in cold)
    assert any(item.tmdb_id > 200 for item, _ in cold)


async def test_artifact_cache_separates_seed_ratings_and_normal_mode(tmp_path, monkeypatch):
    model().save(tmp_path / "model", {"strategy": "item_knn", "model_version": "test"})
    engine = ArtifactRecommendationEngine(tmp_path / "model")
    calls = []
    original = engine._explore

    def tracked(ratings, limit, seed):
        calls.append(seed)
        return original(ratings, limit, seed)

    monkeypatch.setattr(engine, "_explore", tracked)
    try:
        first = await engine.explore({100: 5}, 10, 0)
        assert first == await engine.explore({100: 5}, 10, 0)
        await engine.explore({100: 4}, 10, 0)
        assert first != await engine.explore({100: 5}, 10, 1)
        assert (await engine.recommend({100: 5}, 10)).mode == "personalized"
        assert calls == [0, 0, 1]
    finally:
        await engine.close()
