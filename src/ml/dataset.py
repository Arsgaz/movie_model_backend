import csv
import hashlib
import io
import json
import math
import os
import tempfile
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

SOURCE_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
SOURCE_SHA256 = "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"


@dataclass(frozen=True, slots=True)
class Film:
    movie_id: int
    tmdb_id: int
    title: str
    genres: str


@dataclass(frozen=True, slots=True)
class Observation:
    user_id: int
    movie_id: int
    rating: float
    timestamp: int


@dataclass(frozen=True)
class Dataset:
    films: tuple[Film, ...]
    ratings: Sequence[Observation]
    manifest: dict


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def download(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "ml-latest-small.zip"
    if target.exists():
        if digest(target) != SOURCE_SHA256:
            raise ValueError("Existing archive checksum differs from the pinned MovieLens release")
        return target
    # A failed download never replaces a valid archive.
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".download")
    try:
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            while block := response.read(1024 * 1024):
                output.write(block)
        if digest(Path(temporary)) != SOURCE_SHA256:
            raise ValueError(
                "MovieLens checksum changed; review the upstream release before updating the pin"
            )
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def _rows(archive: zipfile.ZipFile, name: str, columns: list[str]) -> list[dict[str, str]]:
    member = archive.getinfo(f"ml-latest-small/{name}")
    if member.file_size > 25_000_000:
        raise ValueError("Unexpectedly large dataset member")
    with archive.open(member) as stream:
        reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8"))
        if reader.fieldnames != columns:
            raise ValueError(f"Unexpected columns in {name}")
        return list(reader)


def prepare(archive_path: Path, destination: Path) -> dict:
    if digest(archive_path) != SOURCE_SHA256:
        raise ValueError("Archive does not match the pinned MovieLens release")
    with zipfile.ZipFile(archive_path) as archive:
        movies = _rows(archive, "movies.csv", ["movieId", "title", "genres"])
        links = _rows(archive, "links.csv", ["movieId", "imdbId", "tmdbId"])
        ratings = _rows(archive, "ratings.csv", ["userId", "movieId", "rating", "timestamp"])
        license_text = archive.read("ml-latest-small/README.txt").decode("utf-8")
    movie_ids = [int(row["movieId"]) for row in movies]
    link_ids = [int(row["movieId"]) for row in links]
    if len(set(movie_ids)) != len(movie_ids) or len(set(link_ids)) != len(link_ids):
        raise ValueError("Duplicate MovieLens movie identity")
    if set(movie_ids) != set(link_ids) or any(movie_id <= 0 for movie_id in movie_ids):
        raise ValueError("Inconsistent movie/link identities")
    mapping = {int(row["movieId"]): int(row["tmdbId"]) for row in links if row["tmdbId"]}
    if any(value <= 0 for value in mapping.values()):
        raise ValueError("Invalid TMDB identity")
    counts = Counter(mapping.values())
    # Ambiguous links are excluded, never silently merged into one runtime movie.
    usable = {key: value for key, value in mapping.items() if counts[value] == 1}
    films = sorted(
        (
            Film(int(row["movieId"]), usable[int(row["movieId"])], row["title"], row["genres"])
            for row in movies
            if int(row["movieId"]) in usable
        ),
        key=lambda film: film.movie_id,
    )
    valid_movie_ids = set(movie_ids)
    observations, seen = [], set()
    for row in ratings:
        user, movie = int(row["userId"]), int(row["movieId"])
        value, timestamp = float(row["rating"]), int(row["timestamp"])
        if (user, movie) in seen:
            raise ValueError("Duplicate user/movie rating")
        seen.add((user, movie))
        if (
            user <= 0
            or movie not in valid_movie_ids
            or timestamp <= 0
            or not math.isfinite(value)
            or not 0.5 <= value <= 5
            or not (value * 2).is_integer()
        ):
            raise ValueError("Invalid rating row")
        if movie in usable:
            observations.append(Observation(user, movie, value, timestamp))
    observations.sort(key=lambda row: (row.timestamp, row.user_id, row.movie_id))
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "movies.json", [asdict(film) for film in films])
    with (destination / "ratings.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["user_id", "movie_id", "rating", "timestamp"])
        writer.writerows((row.user_id, row.movie_id, row.rating, row.timestamp) for row in observations)
    (destination / "SOURCE_README.txt").write_text(license_text, encoding="utf-8")
    manifest = {
        "format_version": 1,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "movies": len(films),
        "ratings": len(observations),
        "users": len({row.user_id for row in observations}),
        "excluded_movies_missing_tmdb": len(movies) - len(mapping),
        "excluded_movies_ambiguous_tmdb": len(mapping) - len(usable),
        "excluded_ratings": len(ratings) - len(observations),
        "files": {
            name: digest(destination / name) for name in ["movies.json", "ratings.csv", "SOURCE_README.txt"]
        },
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def load_dataset(directory: Path) -> Dataset:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] == 2:
        from ml.large_dataset import load_large_dataset

        return load_large_dataset(directory, manifest)
    if manifest["format_version"] != 1 or manifest["source_sha256"] != SOURCE_SHA256:
        raise ValueError("Unsupported prepared dataset")
    for name in ("movies.json", "ratings.csv", "SOURCE_README.txt"):
        if digest(directory / name) != manifest["files"][name]:
            raise ValueError(f"Prepared data checksum mismatch: {name}")
    films = tuple(Film(**row) for row in json.loads((directory / "movies.json").read_text(encoding="utf-8")))
    with (directory / "ratings.csv").open(encoding="utf-8", newline="") as stream:
        ratings = tuple(
            Observation(
                int(row["user_id"]), int(row["movie_id"]), float(row["rating"]), int(row["timestamp"])
            )
            for row in csv.DictReader(stream)
        )
    return Dataset(films, ratings, manifest)
