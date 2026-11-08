import asyncio
import math
import random
import re
from collections import Counter
from dataclasses import replace

from domain.value_objects.media_type import MediaType
from port.content_recommendation import ContentCandidates, MoviePreference
from port.exceptions.movie_metadata import (
    MovieMetadataNotConfiguredError,
    MovieMetadataNotFoundError,
    MovieMetadataUnavailableError,
)
from port.movie_metadata import MovieMetadata, MovieMetadataProvider
from port.recommendation import RankedMovie

METADATA_ERRORS = (MovieMetadataNotConfiguredError, MovieMetadataNotFoundError, MovieMetadataUnavailableError)
# This is lexical overlap, not semantic embeddings. Genres work even for empty descriptions.
STOP_WORDS = frozenset(
    "который которая которые этого после чтобы когда только своих жизнь фильм история "
    "the and with that this from their have into about".split()
)


def tokens(text: str) -> frozenset[str]:
    return frozenset(word for word in re.findall(r"[^\W\d_]{4,}", text.casefold()) if word not in STOP_WORDS)


def features(movie: MovieMetadata) -> tuple[frozenset[int], frozenset[str]]:
    return frozenset(genre.tmdb_id for genre in movie.genres), tokens(movie.overview)


def similarity(
    left: tuple[frozenset[int], frozenset[str]], right: tuple[frozenset[int], frozenset[str]]
) -> float:
    a, x = left
    b, y = right
    genre_score = len(a & b) / len(a | b) if a or b else 0
    text_score = len(x & y) / math.sqrt(len(x) * len(y)) if x and y else 0
    return 0.8 * genre_score + 0.2 * text_score


class MetadataContentRecommender:
    """A bounded, untrained content heuristic over fresh TMDB discovery pages."""

    def __init__(self, metadata: MovieMetadataProvider) -> None:
        self._metadata = metadata

    async def _pages(self, requests: tuple[tuple[int | None, int], ...]) -> tuple[list[MovieMetadata], bool]:
        async def page(genre: int | None, number: int):
            return await self._metadata.get_page(
                media_type=MediaType.MOVIE,
                query="",
                genre_id=genre,
                page=number,
            )

        results = await asyncio.gather(
            *(page(genre, number) for genre, number in requests), return_exceptions=True
        )
        movies: dict[int, MovieMetadata] = {}
        successful = 0
        for result in results:
            if isinstance(result, METADATA_ERRORS):
                continue
            if isinstance(result, BaseException):
                raise result
            successful += 1
            for movie in result.items:
                if movie.media_type == MediaType.MOVIE:
                    movies.setdefault(movie.tmdb_id, movie)
        if not successful:
            raise MovieMetadataUnavailableError
        return list(movies.values()), successful != len(requests)

    @staticmethod
    def _candidate(movie: MovieMetadata, score: float) -> RankedMovie:
        return RankedMovie(
            movie.tmdb_id, movie.title, any(g.tmdb_id == 16 for g in movie.genres), score, movie
        )

    async def recommend(self, preferences: tuple[MoviePreference, ...], limit: int) -> ContentCandidates:
        # Only the last 50 ratings are seeds; all ratings remain in the exclusion set.
        seeds = preferences[:50]
        positive = [entry for entry in seeds if entry.value >= 4]
        if not positive:
            return ContentCandidates(())
        genres: Counter[int] = Counter()
        for entry in positive:
            for genre in entry.movie.genres:
                genres[genre.tmdb_id] += entry.value - 3
        selected = sorted(genres, key=lambda genre: (-genres[genre], genre))[:2]
        pages = range(1, 3 if limit > 40 else 2)
        requests = tuple((genre, page) for genre in (selected or [None]) for page in pages)
        movies, partial = await self._pages(requests)
        excluded = {entry.movie.tmdb_id for entry in preferences}
        prepared_seeds = [(features(entry.movie), entry.value - 3) for entry in seeds]
        candidates = []
        for movie in movies:
            if movie.tmdb_id in excluded:
                continue
            candidate_features = features(movie)
            score = sum(weight * similarity(seed, candidate_features) for seed, weight in prepared_seeds)
            if score > 0:
                candidates.append(self._candidate(movie, score))
        candidates.sort(key=lambda item: (-item.score, -item.metadata.popularity, item.tmdb_id))
        return ContentCandidates(tuple(candidates[:limit]), partial)

    async def popular(self, excluded: frozenset[int], limit: int) -> ContentCandidates:
        pages = range(1, 4 if limit > 40 else 3)
        movies, partial = await self._pages(tuple((None, page) for page in pages))
        movies.sort(key=lambda movie: (-movie.popularity, movie.tmdb_id))
        return ContentCandidates(
            tuple(
                self._candidate(movie, movie.popularity) for movie in movies if movie.tmdb_id not in excluded
            )[:limit],
            partial,
        )

    async def explore(
        self, preferences: tuple[MoviePreference, ...], limit: int, seed: int, year: int
    ) -> ContentCandidates:
        rng = random.Random(seed)
        positive = [entry for entry in preferences[:50] if entry.value >= 4]
        genres = sorted({genre.tmdb_id for entry in positive for genre in entry.movie.genres})
        requests = []
        for index in range(4):
            genre = rng.choice(genres) if genres and index < 2 else None
            selected_year = rng.randint(1900, year)
            if genre is not None:
                dated = [entry.movie.release_date for entry in positive if entry.movie.release_date]
                if dated:
                    center = min(year, max(1900, rng.choice(dated).year))
                    selected_year = rng.randint(max(1900, center - 15), min(year, center + 15))
            requests.append((genre, selected_year, rng.random()))

        async def page(genre, selected_year, fraction):
            first = await self._metadata.get_exploration_page(genre_id=genre, year=selected_year, page=1)
            number = 1 + int(fraction * min(first.total_pages, 500)) if first.total_pages else 1
            number = min(number, max(1, min(first.total_pages, 500)))
            return (
                first
                if number == 1
                else await self._metadata.get_exploration_page(
                    genre_id=genre, year=selected_year, page=number
                )
            )

        results = await asyncio.gather(*(page(*request) for request in requests), return_exceptions=True)
        movies = {}
        successes = 0
        for result in results:
            if isinstance(result, METADATA_ERRORS):
                continue
            if isinstance(result, BaseException):
                raise result
            successes += 1
            for movie in result.items:
                if movie.media_type == MediaType.MOVIE:
                    movies.setdefault(movie.tmdb_id, movie)
        if not successes:
            raise MovieMetadataUnavailableError
        excluded = {entry.movie.tmdb_id for entry in preferences}
        prepared = [(features(entry.movie), entry.value - 3) for entry in preferences[:50]]
        candidates = []
        for identity, movie in sorted(movies.items()):
            if identity in excluded:
                continue
            score = sum(
                weight * similarity(seed_features, features(movie)) for seed_features, weight in prepared
            )
            if score >= 0:
                candidates.append(replace(self._candidate(movie, score), exploratory=score == 0))
        # The service performs the final weighted sampling and taste/exploration balance.
        rng.shuffle(candidates)
        return ContentCandidates(tuple(candidates[:limit]), successes != len(requests))
