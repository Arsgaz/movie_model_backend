"""Streaming preparation and disk-backed access for the pinned MovieLens 32M release."""

import csv
import io
import json
import math
import os
import tempfile
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ml.dataset import Dataset, Film, Observation, digest, write_json

SOURCE_URL_32M = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
SOURCE_SHA256_32M = "e4a68655d7386b8f95f2f2424b2ff975dfdd15ffd59e0d864a14dca43e99d6ee"
RATING_DTYPE = np.dtype([("user", "<i4"), ("movie", "<i4"), ("value", "<f4"), ("time", "<i8")])


class DiskRatings(Sequence[Observation]):
    def __init__(self, data: np.ndarray, indices: np.ndarray | None = None) -> None:
        self.data, self.indices = data, indices

    def __len__(self) -> int:
        return len(self.data) if self.indices is None else len(self.indices)

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = np.arange(len(self.data), dtype=np.int32) if self.indices is None else self.indices
            return DiskRatings(self.data, indices[key])
        row = self.data[key if self.indices is None else self.indices[key]]
        return Observation(int(row["user"]), int(row["movie"]), float(row["value"]), int(row["time"]))

    def column(self, name: str) -> np.ndarray:
        column = self.data[name]
        return column if self.indices is None else column[self.indices]

    def before(self, timestamp: int) -> "DiskRatings":
        return self.between(None, timestamp)

    def between(self, start: int | None, stop: int | None) -> "DiskRatings":
        time = self.column("time")
        mask = np.ones(len(time), dtype=bool)
        if start is not None:
            mask &= time >= start
        if stop is not None:
            mask &= time < stop
        positions = np.flatnonzero(mask).astype(np.int32)
        return DiskRatings(self.data, positions if self.indices is None else self.indices[positions])


def download_large(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "ml-32m.zip"
    if target.exists():
        if digest(target) != SOURCE_SHA256_32M:
            raise ValueError("MovieLens 32M checksum mismatch")
        return target
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".download")
    try:
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(SOURCE_URL_32M, timeout=60) as response:
            while block := response.read(1024 * 1024):
                output.write(block)
        if digest(Path(temporary)) != SOURCE_SHA256_32M:
            raise ValueError("MovieLens 32M checksum changed; review the release before changing the pin")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def _reader(archive, name, columns):
    stream = io.TextIOWrapper(archive.open(f"ml-32m/{name}"), encoding="utf-8", newline="")
    reader = csv.reader(stream)
    if next(reader) != columns:
        stream.close()
        raise ValueError(f"Unexpected columns in {name}")
    return stream, reader


def prepare_large(archive_path: Path, destination: Path) -> dict:
    if destination.exists():
        raise ValueError("Prepared dataset already exists; choose another output directory")
    if digest(archive_path) != SOURCE_SHA256_32M:
        raise ValueError("Archive does not match the pinned MovieLens 32M release")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".prepare-32m-") as temporary:
        work = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            stream, rows = _reader(archive, "movies.csv", ["movieId", "title", "genres"])
            with stream:
                movie_rows = list(rows)
                movies = {int(row[0]): (row[1], row[2]) for row in movie_rows}
                if len(movie_rows) != len(movies):
                    raise ValueError("Duplicate movie identity")
            stream, rows = _reader(archive, "links.csv", ["movieId", "imdbId", "tmdbId"])
            with stream:
                link_rows = list(rows)
                links = {int(row[0]): int(row[2]) if row[2] else None for row in link_rows}
                if len(link_rows) != len(links):
                    raise ValueError("Duplicate link identity")
                if any(value is not None and not 0 < value < 2**31 for value in links.values()):
                    raise ValueError("Invalid TMDB identity")
            if set(movies) != set(links) or any(not 0 < key < 2**31 for key in movies):
                raise ValueError("Inconsistent movie identities")
            counts = Counter(value for value in links.values() if value is not None)
            usable = {
                key: value for key, value in links.items() if value and value > 0 and counts[value] == 1
            }
            films = [Film(key, value, *movies[key]) for key, value in sorted(usable.items())]
            write_json(work / "movies.json", [asdict(film) for film in films])
            (work / "SOURCE_README.txt").write_bytes(archive.read("ml-32m/README.txt"))
            stream, rows = _reader(archive, "ratings.csv", ["userId", "movieId", "rating", "timestamp"])
            previous = (0, 0)
            users, count, raw_count, minimum, maximum = set(), 0, 0, 2**63 - 1, 0
            batch = []
            with stream, (work / "ratings.bin").open("wb") as output:
                for row in rows:
                    user, movie, value, timestamp = int(row[0]), int(row[1]), float(row[2]), int(row[3])
                    # The official file is ordered by user, then movie; enforce uniqueness without a huge set.
                    if (user, movie) <= previous:
                        raise ValueError("Ratings must be strictly ordered by user/movie without duplicates")
                    previous = (user, movie)
                    if (
                        not 0 < user < 2**31
                        or movie not in movies
                        or not 0 < timestamp < 2**63
                        or not math.isfinite(value)
                        or not 0.5 <= value <= 5
                        or not (value * 2).is_integer()
                    ):
                        raise ValueError("Invalid rating row")
                    raw_count += 1
                    if movie in usable:
                        batch.append((user, movie, value, timestamp))
                        count += 1
                        users.add(user)
                        minimum, maximum = min(minimum, timestamp), max(maximum, timestamp)
                    if len(batch) == 100_000:
                        output.write(np.array(batch, dtype=RATING_DTYPE).tobytes())
                        batch.clear()
                    if raw_count % 2_000_000 == 0:
                        print(f"Prepared {raw_count:,} source ratings", flush=True)
                if batch:
                    output.write(np.array(batch, dtype=RATING_DTYPE).tobytes())
            manifest = {
                "format_version": 2,
                "source": "32m",
                "source_url": SOURCE_URL_32M,
                "source_sha256": SOURCE_SHA256_32M,
                "movies": len(films),
                "ratings": count,
                "source_ratings": raw_count,
                "users": len(users),
                "excluded_movies_missing_tmdb": sum(value is None for value in links.values()),
                "excluded_movies_ambiguous_tmdb": sum(
                    value is not None and counts[value] > 1 for value in links.values()
                ),
                "excluded_ratings": raw_count - count,
                "min_timestamp": minimum,
                "max_timestamp": maximum,
                "rating_dtype": "int32_user,int32_movie,float32_value,int64_timestamp_little_endian",
                "files": {
                    name: digest(work / name) for name in ("movies.json", "ratings.bin", "SOURCE_README.txt")
                },
            }
            write_json(work / "manifest.json", manifest)
        # Publish only a completely written, validated dataset.
        work.rename(destination)
    return manifest


def load_large_dataset(directory: Path, manifest: dict) -> Dataset:
    if manifest["source_sha256"] != SOURCE_SHA256_32M:
        raise ValueError("Unsupported large dataset source")
    for name in ("movies.json", "ratings.bin", "SOURCE_README.txt"):
        if digest(directory / name) != manifest["files"][name]:
            raise ValueError(f"Prepared data checksum mismatch: {name}")
    if (directory / "ratings.bin").stat().st_size != manifest["ratings"] * RATING_DTYPE.itemsize:
        raise ValueError("Invalid ratings file size")
    films = tuple(Film(**row) for row in json.loads((directory / "movies.json").read_text(encoding="utf-8")))
    data = np.memmap(directory / "ratings.bin", dtype=RATING_DTYPE, mode="r")
    return Dataset(films, DiskRatings(data), manifest)
