"""Full-dataset descriptive analysis; no fitting, inference or application database access."""

import csv
import math
import platform
import re
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ml.dataset import Dataset, digest, load_dataset, write_json
from ml.large_dataset import RATING_DTYPE, DiskRatings

QUANTILES = {"p25": 0.25, "median": 0.5, "p75": 0.75, "p90": 0.9, "p95": 0.95, "p99": 0.99}


@dataclass
class DatasetAnalysis:
    summary: dict
    user_counts: Counter
    movie_counts: np.ndarray
    movie_sums: np.ndarray
    positive_counts: np.ndarray
    rating_counts: np.ndarray
    months: dict[int, tuple[int, float, int]]
    genres: list[dict]
    years: list[dict]
    concentration: list[dict]


def rating_chunks(dataset: Dataset, chunk_size: int) -> Iterator[np.ndarray]:
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive")
    ratings = dataset.ratings
    for start in range(0, len(ratings), chunk_size):
        stop = min(start + chunk_size, len(ratings))
        if isinstance(ratings, DiskRatings):
            yield (
                ratings.data[start:stop]
                if ratings.indices is None
                else ratings.data[ratings.indices[start:stop]]
            )
        else:
            yield np.array(
                [(row.user_id, row.movie_id, row.rating, row.timestamp) for row in ratings[start:stop]],
                dtype=RATING_DTYPE,
            )


def describe_counts(values: np.ndarray) -> dict:
    if not len(values):
        return {"count": 0, "min": None, "max": None, "mean": None, **dict.fromkeys(QUANTILES)}
    return {
        "count": len(values),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        **{name: float(np.quantile(values, fraction)) for name, fraction in QUANTILES.items()},
    }


def rating_quantiles(counts: np.ndarray) -> dict:
    """Exact NumPy linear quantiles from ten frequency bins, without expanding 32M values."""
    cumulative = np.cumsum(counts)
    n = int(cumulative[-1])
    if n < 1:
        raise ValueError("Rating quantiles need at least one observation")
    result = {}
    for name, fraction in QUANTILES.items():
        rank = (n - 1) * fraction
        low, high = math.floor(rank), math.ceil(rank)
        a, b = np.searchsorted(cumulative, [low, high], side="right") / 2
        result[name] = float(a + (b - a) * (rank - low))
    return result


def gini(values: np.ndarray) -> float:
    ordered = np.sort(values.astype(np.float64))
    if not len(ordered) or not ordered.sum():
        return 0.0
    value = 2 * np.dot(np.arange(1, len(ordered) + 1), ordered) / (len(ordered) * ordered.sum())
    return float(np.clip(value - (len(ordered) + 1) / len(ordered), 0, 1))


def release_year(title: str) -> int | None:
    match = re.search(r"\((\d{4})\)\s*$", title)
    return int(match[1]) if match and 1850 <= int(match[1]) <= 2100 else None


def collect_statistics(dataset: Dataset, *, chunk_size: int = 1_000_000) -> DatasetAnalysis:
    if not len(dataset.ratings) or not dataset.films:
        raise ValueError("Analysis needs a nonempty prepared dataset")
    ids = np.array([film.movie_id for film in dataset.films], dtype=np.int64)
    if len(np.unique(ids)) != len(ids) or np.any(np.diff(ids) <= 0):
        raise ValueError("Prepared movie IDs must be unique and sorted")
    users: Counter = Counter()
    counts = np.zeros(len(ids), dtype=np.int64)
    sums = np.zeros(len(ids), dtype=np.float64)
    positives = np.zeros(len(ids), dtype=np.int64)
    ratings = np.zeros(11, dtype=np.int64)
    months: dict[int, tuple[int, float, int]] = {}
    first, last, processed = 2**63 - 1, 0, 0
    for batch in rating_chunks(dataset, chunk_size):
        values, timestamps = batch["value"], batch["time"]
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.5)
            or np.any(values > 5)
            or np.any(values * 2 != np.floor(values * 2))
            or np.any(batch["user"] <= 0)
            or np.any(timestamps <= 0)
        ):
            raise ValueError("Invalid prepared rating")
        indices = np.searchsorted(ids, batch["movie"])
        if np.any(indices >= len(ids)) or np.any(ids[indices] != batch["movie"]):
            raise ValueError("Rating references an unknown prepared movie")
        counts += np.bincount(indices, minlength=len(ids))
        sums += np.bincount(indices, weights=values, minlength=len(ids))
        positives += np.bincount(indices[values >= 4], minlength=len(ids))
        ratings += np.bincount((values * 2).astype(np.int8), minlength=11)
        unique, frequency = np.unique(batch["user"], return_counts=True)
        users.update(dict(zip(unique.tolist(), frequency.tolist(), strict=True)))
        calendar = timestamps.astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)
        unique_months, inverse, frequency = np.unique(calendar, return_inverse=True, return_counts=True)
        monthly_sums = np.bincount(inverse, weights=values)
        monthly_positives = np.bincount(inverse, weights=values >= 4).astype(np.int64)
        for index, month in enumerate(unique_months):
            previous = months.get(int(month), (0, 0.0, 0))
            months[int(month)] = (
                previous[0] + int(frequency[index]),
                previous[1] + float(monthly_sums[index]),
                previous[2] + int(monthly_positives[index]),
            )
        first, last = min(first, int(timestamps.min())), max(last, int(timestamps.max()))
        processed += len(batch)
    if processed != len(dataset.ratings):
        raise ValueError("Incomplete analysis")
    months = {month: months.get(month, (0, 0.0, 0)) for month in range(min(months), max(months) + 1)}
    user_values = np.fromiter(users.values(), dtype=np.int64)
    rated = counts > 0
    total = int(counts.sum())
    density = total / (len(users) * len(ids))
    if density > 1:
        raise ValueError("More observations than user/movie pairs; prepared data is not unique")
    nonempty = np.flatnonzero(ratings)
    genre_records: dict[str, list[int]] = {}
    year_records: dict[int | None, list[int]] = {}
    missing_genres = 0
    for index, film in enumerate(dataset.films):
        labels = {label for label in film.genres.split("|") if label and label != "(no genres listed)"}
        if not labels:
            missing_genres += 1
        for genre in labels:
            record = genre_records.setdefault(genre, [0, 0, 0])
            record[0] += 1
            record[1] += int(counts[index])
            record[2] += int(positives[index])
        year = year_records.setdefault(release_year(film.title), [0, 0])
        year[0] += 1
        year[1] += int(counts[index])
    genres = [
        {"genre": label, "movies": values[0], "ratings": values[1], "positive_ratings": values[2]}
        for label, values in sorted(genre_records.items())
    ]
    years = [
        {"year": year, "movies": values[0], "ratings": values[1]}
        for year, values in sorted(year_records.items(), key=lambda pair: pair[0] or 0)
    ]
    concentration = []
    for population, values in (("users", user_values), ("movies", counts)):
        ordered = np.sort(values)[::-1]
        for fraction in (0.01, 0.05, 0.1, 0.2, 0.5, 1):
            number = math.ceil(len(values) * fraction)
            concentration.append(
                {
                    "population": population,
                    "requested_top_fraction": fraction,
                    "entities": number,
                    "actual_top_fraction": number / len(values),
                    "ratings": int(ordered[:number].sum()),
                    "rating_share": float(ordered[:number].sum() / total),
                }
            )
    source = dataset.manifest
    summary = {
        "ratings": total,
        "users": len(users),
        "catalog_movies": len(ids),
        "rated_movies": int(rated.sum()),
        "movies_without_ratings": int((~rated).sum()),
        "density": density,
        "sparsity": 1 - density,
        "rated_catalog_density": total / (len(users) * int(rated.sum())),
        "rated_catalog_sparsity": 1 - total / (len(users) * int(rated.sum())),
        "rating_values": {
            "min": float(nonempty[0] / 2),
            "max": float(nonempty[-1] / 2),
            "mean": float(sums.sum() / total),
            **rating_quantiles(ratings),
        },
        "positive_ratings": int(positives.sum()),
        "positive_rating_share": float(positives.sum() / total),
        "user_activity": describe_counts(user_values),
        "rated_movie_activity": describe_counts(counts[rated]),
        "user_activity_gini": gini(user_values),
        "movie_activity_gini": gini(counts),
        "first_rating_utc": datetime.fromtimestamp(first, UTC).isoformat(),
        "last_rating_utc": datetime.fromtimestamp(last, UTC).isoformat(),
        "movies_without_genres": missing_genres,
        "movies_without_release_year": year_records.get(None, [0, 0])[0],
        "source_quality": {
            name: source.get(name)
            for name in (
                "source_ratings",
                "excluded_movies_missing_tmdb",
                "excluded_movies_ambiguous_tmdb",
                "excluded_ratings",
            )
        },
        "analysis_scope": "entire_prepared_dataset_no_sampling",
        "unique_pairs": "guaranteed_by_preparation_not_rechecked_as_a_full_pair_set",
        "notes": [
            "Genres are nonexclusive; totals by genre may exceed the catalog and rating counts.",
            "Activity includes full available histories; it is descriptive, not a training feature.",
            "Release years are parsed from movie titles, separately from rating timestamps.",
            "Boundary calendar months can be incomplete.",
            "Unobserved user/movie pairs are missing feedback, not negative ratings.",
        ],
    }
    return DatasetAnalysis(
        summary, users, counts, sums, positives, ratings, months, genres, years, concentration
    )


def write_csv(path: Path, columns: list[str], rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)


def analyze(dataset_path: Path, output: Path, *, chunk_size: int = 1_000_000) -> dict:
    if chunk_size < 1 or output.exists():
        raise ValueError("Use a positive chunk size and a new output directory")
    dataset = load_dataset(dataset_path)
    result = collect_statistics(dataset, chunk_size=chunk_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".analysis-") as temporary:
        work = Path(temporary)
        write_json(work / "summary.json", result.summary)
        write_csv(
            work / "rating_distribution.csv",
            ["rating", "count", "share"],
            (
                (
                    i / 2,
                    int(result.rating_counts[i]),
                    float(result.rating_counts[i] / result.summary["ratings"]),
                )
                for i in range(1, 11)
            ),
        )
        write_csv(
            work / "user_activity.csv", ["movielens_user_id", "ratings"], sorted(result.user_counts.items())
        )
        write_csv(
            work / "movies.csv",
            [
                "movielens_id",
                "tmdb_id",
                "title",
                "genres",
                "release_year",
                "ratings",
                "mean_rating",
                "positive_ratings",
            ],
            (
                (
                    film.movie_id,
                    film.tmdb_id,
                    film.title,
                    film.genres,
                    release_year(film.title),
                    int(result.movie_counts[i]),
                    float(result.movie_sums[i] / result.movie_counts[i]) if result.movie_counts[i] else None,
                    int(result.positive_counts[i]),
                )
                for i, film in enumerate(dataset.films)
            ),
        )
        write_csv(
            work / "monthly_ratings.csv",
            ["month_utc", "ratings", "mean_rating", "positive_ratings"],
            (
                (
                    str(np.datetime64(month, "M")),
                    values[0],
                    values[1] / values[0] if values[0] else None,
                    values[2],
                )
                for month, values in sorted(result.months.items())
            ),
        )
        for name, rows in (
            ("genres", result.genres),
            ("release_years", result.years),
            ("concentration", result.concentration),
        ):
            columns = (
                list(rows[0])
                if rows
                else (["genre", "movies", "ratings", "positive_ratings"] if name == "genres" else [])
            )
            write_csv(work / f"{name}.csv", columns, (record.values() for record in rows))
        from ml.analysis_plots import render_plots

        version = render_plots(result, work)
        manifest = {
            "format_version": 1,
            "artifact": "dataset_analysis",
            "source_sha256": dataset.manifest["source_sha256"],
            "prepared_manifest_sha256": digest(dataset_path / "manifest.json"),
            "chunk_size": chunk_size,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": version},
            "files": {path.name: digest(path) for path in sorted(work.iterdir()) if path.is_file()},
        }
        write_json(work / "manifest.json", manifest)
        work.rename(output)
    return result.summary
