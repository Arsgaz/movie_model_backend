import json
import zipfile

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from ml.dataset import Film, Observation, digest, load_dataset  # noqa: E402
from ml.evaluation import chronological_split, evaluate  # noqa: E402
from ml.large_dataset import RATING_DTYPE, DiskRatings, prepare_large  # noqa: E402
from ml.large_evaluation import evaluate_large  # noqa: E402
from ml.model import ItemModel, Parameters  # noqa: E402
from ml.pipeline import train  # noqa: E402


def fixture_archive(tmp_path, monkeypatch, duplicate=False):
    import ml.large_dataset as module

    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "ml-32m/movies.csv",
            "movieId,title,genres\n" + "".join(f"{i},Film {i},Drama\n" for i in range(1, 11)),
        )
        output.writestr(
            "ml-32m/links.csv",
            "movieId,imdbId,tmdbId\n" + "".join(f"{i},1,{i * 100 if i < 10 else ''}\n" for i in range(1, 11)),
        )
        rows = "".join(
            f"{user},{movie},5,{movie * 100 + user}\n" for user in range(1, 5) for movie in range(1, 11)
        )
        output.writestr(
            "ml-32m/ratings.csv",
            "userId,movieId,rating,timestamp\n" + rows + ("4,10,5,1004\n" if duplicate else ""),
        )
        output.writestr("ml-32m/README.txt", "Fixture license")
    monkeypatch.setattr(module, "SOURCE_SHA256_32M", digest(archive))
    return archive


def test_streaming_prepare_roundtrip_checksums_and_no_working_database(tmp_path, monkeypatch):
    archive = fixture_archive(tmp_path, monkeypatch)
    prepared = tmp_path / "prepared"
    manifest = prepare_large(archive, prepared)
    assert manifest["source_ratings"] == 40 and manifest["ratings"] == 36
    assert manifest["excluded_ratings"] == 4
    dataset = load_dataset(prepared)
    assert isinstance(dataset.ratings, DiskRatings)
    assert isinstance(dataset.ratings.data, np.memmap)
    assert dataset.ratings[0] == Observation(1, 1, 5, 101)
    assert dataset.ratings[-1] == Observation(4, 9, 5, 904)
    assert not list(tmp_path.glob("*.sqlite*"))
    with pytest.raises(ValueError, match="already exists"):
        prepare_large(archive, prepared)
    with (prepared / "ratings.bin").open("r+b") as output:
        output.write(b"wrong bytes")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(prepared)


def test_duplicate_source_ratings_do_not_publish_partial_dataset(tmp_path, monkeypatch):
    archive = fixture_archive(tmp_path, monkeypatch, duplicate=True)
    with pytest.raises(ValueError, match="duplicates"):
        prepare_large(archive, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()
    assert not list(tmp_path.glob(".prepare-32m-*"))


def test_disk_split_keeps_ties_and_global_time_boundaries():
    rows = np.array(
        [(user, movie, 5, movie * 100 + user) for user in range(1, 5) for movie in range(1, 11)],
        dtype=RATING_DTYPE,
    )
    data = DiskRatings(rows)
    parts = chronological_split(data)
    expected = chronological_split(tuple(data))
    assert all(isinstance(part, DiskRatings) for part in parts)
    assert [set(part) for part in parts] == [set(part) for part in expected]
    assert max(parts[0].column("time")) < min(parts[1].column("time"))
    assert max(parts[1].column("time")) < min(parts[2].column("time"))
    assert len(parts[0]) + len(parts[1]) + len(parts[2]) == len(data)


def test_blockwise_neighbors_equal_dense_reference():
    from scipy import sparse

    films = [Film(i, i * 100, str(i), "Drama") for i in range(1, 8)]
    rows = [
        Observation(user, movie, 5 if (movie + user) % 3 else 1, user * 100 + movie)
        for user in range(1, 6)
        for movie in range(1, 8)
    ]
    parameters = Parameters(neighbors=3, shrinkage=10, min_support=1)
    model = ItemModel.fit(films, rows, parameters, block_size=1)
    other = ItemModel.fit(films, rows, parameters, block_size=4)
    np.testing.assert_array_equal(model.similarities.toarray(), other.similarities.toarray())
    matrix = np.zeros((6, 7))
    for row in rows:
        matrix[row.user_id, row.movie_id - 1] = row.rating >= 4
    support = matrix.T @ matrix
    norm = np.sqrt(matrix.sum(axis=0))
    reference = np.zeros((7, 7))
    for i in range(7):
        values = support[i] / (norm[i] * norm) * support[i] / (support[i] + 10)
        values[i] = 0
        order = sorted((j for j in range(7) if support[i, j] >= 1 and j != i), key=lambda j: (-values[j], j))[
            :3
        ]
        reference[i, order] = values[order]
    np.testing.assert_allclose(model.similarities.toarray(), reference, atol=1e-7)
    assert sparse.isspmatrix_csr(model.similarities)


def test_recent_positive_cap_does_not_truncate_popularity_or_seen_exclusions():
    films = [Film(i, i * 100, str(i), "Drama") for i in range(1, 4)]
    rows = [Observation(user, movie, 5, movie) for user in (1, 2) for movie in (1, 2, 3)]
    model = ItemModel.fit(films, rows, Parameters(min_support=1, max_user_positives=2))
    assert model.popularity.tolist() == [2, 2, 2]
    assert model.similarities.getrow(0).nnz == 0
    assert model.similarities[1, 2] > 0
    items, _ = model.recommend({100: 5, 200: 5})
    assert [item.tmdb_id for item in items] == [300]


def test_large_evaluation_matches_full_cohort_and_samples_reproducibly():
    films = [Film(i, i * 100, str(i), "Drama") for i in range(1, 8)]
    history = tuple(Observation(user, movie, 5, movie) for user in range(1, 8) for movie in range(1, 5))
    targets = tuple(Observation(user, movie, 5, movie) for user in range(1, 8) for movie in range(5, 8))
    model = ItemModel.fit(films, history, Parameters(min_support=1))

    def disk(rows):
        return DiskRatings(
            np.array([(r.user_id, r.movie_id, r.rating, r.timestamp) for r in rows], dtype=RATING_DTYPE)
        )

    expected = evaluate(model, history, targets, films)
    actual = evaluate_large(model, disk(history), disk(targets), films, max_users=100)
    for key in ("precision@10", "recall@10", "ndcg@10", "evaluated_users", "unknown_target_positive_ratings"):
        assert actual[key] == expected[key]
    sampled = evaluate_large(model, disk(history), disk(targets), films, max_users=3)
    assert sampled == evaluate_large(model, disk(history), disk(targets), films, max_users=3)
    assert sampled["evaluated_users"] == 3 and sampled["eligible_users"] == 7


def test_disk_training_pipeline_uses_all_rows_for_final_artifact(tmp_path, monkeypatch):
    archive = fixture_archive(tmp_path, monkeypatch)
    prepare_large(archive, tmp_path / "prepared")
    summary = train(tmp_path / "prepared", tmp_path / "run", evaluation_users=3, block_size=2)
    model, manifest = ItemModel.load(tmp_path / "run/model")
    assert summary["ratings"] == manifest["training_ratings"] == 36
    assert len(model.films) == 9
    metrics = json.loads((tmp_path / "run/metrics.json").read_text())
    assert metrics["evaluation_user_limit"] == 3
    assert metrics["selected_parameters"]["max_user_positives"] == 100
    assert (
        metrics["validation"]["popularity"]["evaluation_users_sha256"]
        == metrics["validation"]["item_knn_trials"][0]["metrics"]["evaluation_users_sha256"]
    )
