import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from ml.bootstrap import paired_bootstrap  # noqa: E402
from ml.dataset import Dataset, Film, digest, write_json  # noqa: E402
from ml.large_dataset import RATING_DTYPE, DiskRatings  # noqa: E402
from ml.large_evaluation import EvaluationCohort  # noqa: E402
from ml.quality_evaluation import (  # noqa: E402
    assess,
    evaluate_cohort,
    history_group,
    item_group,
    summarize_pairs,
)


def test_bootstrap_preserves_pairing_and_known_constant_difference():
    values = np.linspace(0, 0.8, 50)
    result = paired_bootstrap(np.column_stack([values, values + 0.1]), resamples=500, seed=7)
    assert result["popularity"]["mean"] == pytest.approx(0.4)
    for name in ("mean", "low", "high"):
        assert result["paired_difference"][name] == pytest.approx(0.1)
    assert result["paired_difference"]["interval_excludes_zero"]
    assert result == paired_bootstrap(np.column_stack([values, values + 0.1]), resamples=500, seed=7)


def test_equal_strategies_have_zero_difference_and_no_claim_of_improvement():
    result = paired_bootstrap(np.array([[0, 0], [1, 1], [0.5, 0.5]]), resamples=100)
    assert result["paired_difference"] == {"mean": 0, "low": 0, "high": 0, "interval_excludes_zero": False}


@pytest.mark.parametrize("values", [[], [[1]], [[float("nan"), 0]]])
def test_bootstrap_rejects_invalid_observations(values):
    with pytest.raises(ValueError):
        paired_bootstrap(np.asarray(values), resamples=100)


@pytest.mark.parametrize(
    "size,expected",
    [
        (1, "1-19"),
        (19, "1-19"),
        (20, "20-99"),
        (99, "20-99"),
        (100, "100-499"),
        (499, "100-499"),
        (500, "500+"),
    ],
)
def test_history_thresholds_are_fixed(size, expected):
    assert history_group(size) == expected


@pytest.mark.parametrize(
    "support,expected",
    [
        (0, "unseen"),
        (1, "tail_1-49"),
        (49, "tail_1-49"),
        (50, "middle_50-999"),
        (999, "middle_50-999"),
        (1000, "head_1000+"),
    ],
)
def test_item_thresholds_use_historical_counts(support, expected):
    assert item_group(support) == expected


def test_item_segments_do_not_filter_or_rerank_recommendations():
    class Model:
        by_tmdb = {100: 0, 200: 1, 300: 2}

        def recommend(self, profile, **_kwargs):
            return [SimpleNamespace(tmdb_id=200), SimpleNamespace(tmdb_id=300)], "item_knn"

    cohort = EvaluationCohort(
        {1: {100: 5}}, {1: {300, 400}}, {100: 2, 200: 2000, 300: 1}, np.array([300, 400]), {}
    )
    result = evaluate_cohort(Model(), cohort, strategy="item_knn", k=2)
    for row in result["per_user"]:
        assert row["recall"] == 0.5
    tail = next(row for row in result["per_item_group"] if row["item_group"] == "tail_1-49")
    assert tail["recall"] == 1
    assert tail["ndcg"] == pytest.approx(1 / np.log2(3))
    unseen = next(row for row in result["per_item_group"] if row["item_group"] == "unseen")
    assert unseen["recall"] == 0
    assert [row["rank"] for row in result["recommendations"] if row["strategy"] == "selected"] == [1, 2]


def test_incomplete_or_duplicate_user_pairs_are_not_silently_bootstrapped():
    row = {"user_id": 1, "strategy": "selected", "precision": 0.1, "recall": 0.2, "ndcg": 0.3}
    with pytest.raises(ValueError, match="same users"):
        summarize_pairs([row], resamples=100, seed=1)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_pairs([row, row], resamples=100, seed=1)
    assert summarize_pairs([], resamples=100, seed=1)["metrics"] == {}


def test_full_replay_matches_frozen_experiment_and_keeps_source_artifacts(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import ml.pipeline as pipeline
    import ml.quality_evaluation as quality

    films = tuple(Film(movie, movie * 100, f"Film {movie}", "Drama") for movie in range(1, 11))
    rows = np.array(
        [(user, movie, 5, movie * 100 + user) for user in range(1, 5) for movie in range(1, 11)],
        dtype=RATING_DTYPE,
    )
    dataset = Dataset(films, DiskRatings(rows), {"source_sha256": "fixture", "users": 4})
    source = tmp_path / "data"
    source.mkdir()
    write_json(source / "manifest.json", dataset.manifest)
    (source / "SOURCE_README.txt").write_text("Fixture license")
    monkeypatch.setattr(pipeline, "load_dataset", lambda _path: dataset)
    monkeypatch.setattr(quality, "load_dataset", lambda _path: dataset)
    run = tmp_path / "run"
    pipeline.train(source, run, evaluation_users=3, block_size=2)
    before = {str(path.relative_to(run)): digest(path) for path in run.rglob("*") if path.is_file()}
    output = tmp_path / "assessment"
    result = assess(source, run, output, resamples=100, block_size=2)
    frozen = json.loads((run / "metrics.json").read_text())
    assert result["overall"]["users"] == 3
    for metric in ("precision", "recall", "ndcg"):
        assert result["overall"]["metrics"][metric]["selected"]["mean"] == pytest.approx(
            frozen["test"]["selected"][f"{metric}@10"]
        )
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.svg"))) == 4
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert digest(output / name) == expected
    assert before == {str(path.relative_to(run)): digest(path) for path in run.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="new output"):
        assess(source, run, output)
    (run / "metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        assess(source, run, tmp_path / "bad-assessment")
