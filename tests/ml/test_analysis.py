import csv
import json
from datetime import UTC, datetime

import pytest

np = pytest.importorskip("numpy")

from ml.analysis import analyze, collect_statistics, rating_quantiles, release_year  # noqa: E402
from ml.dataset import Dataset, Film, Observation, digest  # noqa: E402
from ml.large_dataset import RATING_DTYPE, DiskRatings  # noqa: E402


def stamp(month):
    return int(datetime(2020, month, 1, tzinfo=UTC).timestamp())


@pytest.fixture
def dataset():
    films = (
        Film(1, 100, "First (2000)", "Drama|Comedy"),
        Film(2, 200, "Second (2001)", "Drama"),
        Film(3, 300, "Unknown year", "(no genres listed)"),
        Film(4, 400, "No ratings (2002)", "Animation"),
    )
    ratings = (
        Observation(1, 1, 1, stamp(1)),
        Observation(1, 2, 0.5, stamp(1)),
        Observation(2, 1, 4, stamp(2)),
        Observation(2, 2, 3, stamp(2)),
        Observation(3, 1, 5, stamp(2)),
        Observation(3, 3, 4.5, stamp(3)),
    )
    return Dataset(films, ratings, {"source_sha256": "fixture"})


def test_statistics_match_hand_calculated_values(dataset):
    result = collect_statistics(dataset, chunk_size=2)
    summary = result.summary
    assert summary["ratings"] == 6 and summary["users"] == 3
    assert summary["catalog_movies"] == 4 and summary["rated_movies"] == 3
    assert summary["movies_without_ratings"] == 1
    assert summary["sparsity"] == 0.5
    assert summary["rated_catalog_sparsity"] == pytest.approx(1 / 3)
    assert summary["rating_values"]["mean"] == 3
    assert summary["rating_values"]["median"] == 3.5
    assert summary["positive_rating_share"] == 0.5
    assert summary["user_activity"]["median"] == 2
    assert summary["user_activity_gini"] == 0
    assert result.movie_counts.tolist() == [3, 2, 1, 0]
    assert result.movie_sums.tolist() == [10, 3.5, 4.5, 0]
    assert result.positive_counts.tolist() == [2, 0, 1, 0]
    assert sum(result.rating_counts) == 6
    assert [value[0] for _, value in sorted(result.months.items())] == [2, 3, 1]
    assert summary["movies_without_genres"] == 1 and summary["movies_without_release_year"] == 1
    drama = next(record for record in result.genres if record["genre"] == "Drama")
    assert drama == {"genre": "Drama", "movies": 2, "ratings": 5, "positive_ratings": 2}
    # Genres are not mutually exclusive; unrated catalog entries stay in the metadata statistics.
    assert sum(record["ratings"] for record in result.genres) == 8
    assert next(record for record in result.years if record["year"] == 2002)["ratings"] == 0
    first = next(row for row in result.concentration if row["population"] == "movies")
    assert first["entities"] == 1 and first["actual_top_fraction"] == 0.25
    assert first["rating_share"] == 0.5


def test_histogram_quantiles_equal_numpy_linear_quantiles():
    values = np.array([0.5, 1, 1, 3, 3, 4.5, 5])
    counts = np.bincount((values * 2).astype(int), minlength=11)
    result = rating_quantiles(counts)
    for key, q in (("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)):
        assert result[key] == pytest.approx(np.quantile(values, q))


def test_chunk_sizes_and_disk_backed_views_produce_identical_statistics(dataset):
    expected = collect_statistics(dataset, chunk_size=1)
    rows = np.array(
        [(r.user_id, r.movie_id, r.rating, r.timestamp) for r in dataset.ratings], dtype=RATING_DTYPE
    )
    # An indexed view must analyze its rows, not the entire mapped file.
    padded = np.concatenate([rows, rows[:1]])
    view = DiskRatings(padded, np.arange(6, dtype=np.int32))
    disk = Dataset(dataset.films, view, dataset.manifest)
    for size in (2, 100):
        actual = collect_statistics(disk, chunk_size=size)
        assert actual.summary == expected.summary
        assert actual.months == expected.months
        assert actual.genres == expected.genres
        assert actual.concentration == expected.concentration


@pytest.mark.parametrize("value", [float("nan"), 0, 6, 3.25])
def test_invalid_ratings_are_not_silently_counted(dataset, value):
    bad = Dataset(dataset.films, (Observation(1, 1, value, stamp(1)),), dataset.manifest)
    with pytest.raises(ValueError, match="Invalid prepared rating"):
        collect_statistics(bad)


def test_unknown_movie_and_empty_dataset_are_rejected(dataset):
    with pytest.raises(ValueError, match="unknown prepared movie"):
        collect_statistics(Dataset(dataset.films, (Observation(1, 99, 5, stamp(1)),), dataset.manifest))
    with pytest.raises(ValueError, match="nonempty"):
        collect_statistics(Dataset(dataset.films, (), dataset.manifest))


@pytest.mark.parametrize(
    "title,year", [("Film (2002)", 2002), ("Film (2002) ", 2002), ("Film (cut)", None), ("Film (3000)", None)]
)
def test_release_year_is_not_inferred_from_rating_date(title, year):
    assert release_year(title) == year


def test_export_writes_standalone_figures_and_verified_tables(dataset, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import ml.analysis as module

    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text('{"fixture": true}')
    monkeypatch.setattr(module, "load_dataset", lambda _path: dataset)
    output = tmp_path / "result"
    summary = analyze(source, output, chunk_size=2)
    assert summary == json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["prepared_manifest_sha256"] == digest(source / "manifest.json")
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.svg"))) == 7
    for name, expected in manifest["files"].items():
        assert digest(output / name) == expected
    with (output / "monthly_ratings.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["month_utc"] for row in rows] == ["2020-01", "2020-02", "2020-03"]
    assert sum(int(row["ratings"]) for row in rows) == 6
    with pytest.raises(ValueError, match="new output directory"):
        analyze(source, output)


def test_failed_plot_export_does_not_publish_partial_artifact(dataset, tmp_path, monkeypatch):
    import ml.analysis as module
    import ml.analysis_plots as plots

    monkeypatch.setattr(module, "load_dataset", lambda _path: dataset)

    def fail(*_args):
        raise ValueError("plotting unavailable")

    monkeypatch.setattr(plots, "render_plots", fail)
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="plotting unavailable"):
        analyze(tmp_path, output)
    assert not output.exists() and not list(tmp_path.glob(".analysis-*"))


def test_missing_calendar_month_has_zero_count_and_no_fabricated_mean(dataset):
    ratings = (Observation(1, 1, 5, stamp(1)), Observation(1, 2, 1, stamp(3)))
    result = collect_statistics(Dataset(dataset.films, ratings, dataset.manifest))
    months = list(sorted(result.months.items()))
    assert len(months) == 3 and months[1][1] == (0, 0.0, 0)
    assert result.summary["ratings"] == 2
