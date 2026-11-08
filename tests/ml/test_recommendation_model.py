import math
from dataclasses import replace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from ml.dataset import SOURCE_SHA256, Film, Observation, load_dataset, prepare  # noqa: E402
from ml.evaluation import chronological_split, evaluate, ranking_metrics  # noqa: E402
from ml.model import ItemModel, Parameters  # noqa: E402


@pytest.fixture
def corpus():
    films = tuple(Film(index, index * 100, f"Film {index}", "Drama") for index in range(1, 6))
    rows = tuple(
        Observation(user, movie, value, user * 10 + movie)
        for user, entries in [
            (1, [(1, 5), (2, 5), (3, 1)]),
            (2, [(1, 4), (2, 5), (4, 5)]),
            (3, [(3, 5), (4, 5), (5, 2)]),
        ]
        for movie, value in entries
    )
    return films, rows


def test_metric_values_are_hand_calculated():
    metrics = ranking_metrics([1, 9, 2], {1, 2, 3}, 3)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["ndcg"] == pytest.approx(1.5 / (1 + 1 / math.log2(3) + 0.5))
    assert ranking_metrics([], {1}, 10) == {"precision": 0, "recall": 0, "ndcg": 0}


def test_global_split_keeps_equal_timestamps_together():
    rows = [Observation(i, 1, 5, timestamp) for i, timestamp in enumerate(range(1, 101), 1)]
    rows[69] = replace(rows[69], timestamp=71)
    train, validation, test = chronological_split(rows)
    assert max(row.timestamp for row in train) < min(row.timestamp for row in validation)
    assert max(row.timestamp for row in validation) < min(row.timestamp for row in test)
    assert set(train) | set(validation) | set(test) == set(rows)
    assert not set(train) & set(validation)


def test_fit_uses_only_training_movies_and_positive_feedback(corpus):
    films, rows = corpus
    model = ItemModel.fit(films, rows[:3], Parameters(min_support=1, shrinkage=0))
    assert set(model.by_tmdb) == {100, 200, 300}
    assert model.similarities[0, 1] == pytest.approx(1)
    assert model.similarities[0, 2] == 0
    assert (model.similarities.diagonal() == 0).all()


def test_new_profile_uses_tmdb_and_excludes_all_rated_movies(corpus):
    films, rows = corpus
    model = ItemModel.fit(films, rows, Parameters(min_support=1))
    items, strategy = model.recommend({100: 5, 300: 1, 90000: 5}, limit=3)
    assert strategy == "item_knn"
    assert items[0].tmdb_id == 200
    assert not {100, 300, 90000} & {item.tmdb_id for item in items}
    assert model.recommend({100: 5, 300: 1}, limit=3) == (items, strategy)


@pytest.mark.parametrize("history", [{}, {90000: 5}, {100: 1}, {100: 3}])
def test_cold_or_negative_only_profiles_use_explicit_popularity(corpus, history):
    films, rows = corpus
    model = ItemModel.fit(films, rows, Parameters())
    items, strategy = model.recommend(history)
    assert strategy == "popularity"
    assert all(item.tmdb_id not in history for item in items)


def test_artifact_round_trip_and_tamper_detection(corpus, tmp_path):
    films, rows = corpus
    model = ItemModel.fit(films, rows, Parameters(min_support=1))
    path = tmp_path / "model"
    model.save(path, {"strategy": "item_knn", "model_version": "test"})
    loaded, manifest = ItemModel.load(path)
    assert loaded.recommend({100: 5}) == model.recommend({100: 5})
    assert manifest["format_version"] == 1
    (path / "movies.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        ItemModel.load(path)


def test_unknown_test_items_count_as_misses_and_cold_users_are_reported(corpus):
    films, rows = corpus
    model = ItemModel.fit(films, rows[:3], Parameters(min_support=1))
    result = evaluate(model, rows[:3], [Observation(1, 4, 5, 100), Observation(99, 2, 5, 100)], films)
    assert result["recall@10"] == 0
    assert result["ndcg@10"] == 0
    assert result["evaluated_users"] == 1
    assert result["skipped_users_without_history"] == 1
    assert result["unknown_target_positive_ratings"] == 1


def test_bad_download_is_rejected_before_preparing(tmp_path):
    path = tmp_path / "bad.zip"
    path.write_bytes(b"wrong release")
    with pytest.raises(ValueError, match="pinned"):
        prepare(path, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_prepared_files_are_checked_before_loading(tmp_path):
    import json

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "source_sha256": SOURCE_SHA256,
                "files": {"movies.json": "incorrect"},
            }
        )
    )
    (tmp_path / "movies.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(tmp_path)


def test_low_rating_reduces_related_candidate_scores(corpus):
    films, rows = corpus
    model = ItemModel.fit(films, rows, Parameters(min_support=1, shrinkage=0))
    positive, _ = model.recommend({100: 5}, limit=5)
    negative, _ = model.recommend({100: 5, 400: 1}, limit=5)
    scores = {item.tmdb_id: item.score for item in positive}
    assert next(item.score for item in negative if item.tmdb_id == 200) < scores[200]


def test_preparation_and_full_training_pipeline_are_reproducible(tmp_path, monkeypatch):
    import json
    import zipfile

    import ml.dataset as dataset_module
    from ml.pipeline import train

    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "ml-latest-small/movies.csv",
            "movieId,title,genres\n" + "".join(f"{i},Film {i},Drama\n" for i in range(1, 11)),
        )
        output.writestr(
            "ml-latest-small/links.csv",
            "movieId,imdbId,tmdbId\n" + "".join(f"{i},1,{i * 100 if i < 10 else ''}\n" for i in range(1, 11)),
        )
        output.writestr(
            "ml-latest-small/ratings.csv",
            "userId,movieId,rating,timestamp\n"
            + "".join(
                f"{user},{movie},5,{movie * 100 + user}\n" for movie in range(1, 11) for user in range(1, 5)
            ),
        )
        output.writestr("ml-latest-small/README.txt", "Fixture license")
    monkeypatch.setattr(dataset_module, "SOURCE_SHA256", dataset_module.digest(archive))
    prepared = tmp_path / "prepared"
    manifest = prepare(archive, prepared)
    assert manifest["ratings"] == 36 and manifest["excluded_ratings"] == 4
    assert manifest["excluded_movies_missing_tmdb"] == 1
    first = train(prepared, tmp_path / "run-a")
    second = train(prepared, tmp_path / "run-b")
    assert first["test"] == second["test"]
    a, _ = ItemModel.load(tmp_path / "run-a/model")
    b, _ = ItemModel.load(tmp_path / "run-b/model")
    assert a.recommend({100: 5}) == b.recommend({100: 5})
    assert a.by_tmdb.keys() == set(range(100, 1000, 100))
    metrics = json.loads((tmp_path / "run-a/metrics.json").read_text())
    assert sum(metrics["split"][key] for key in ("train", "validation", "test")) == 36
    assert metrics["test"]["selected"]["evaluated_users"] > 0
    with pytest.raises(ValueError, match="already exists"):
        train(prepared, tmp_path / "run-a")
