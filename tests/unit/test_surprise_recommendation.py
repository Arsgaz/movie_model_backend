from datetime import UTC, datetime

import pytest

from application.service.content_recommendation import MetadataContentRecommender
from application.service.hybrid_recommendation import RecommendationPolicy
from application.service.surprise_recommendation import SurpriseRecommender
from port.content_recommendation import ContentCandidates, MoviePreference
from port.exceptions.movie_metadata import MovieMetadataUnavailableError
from port.exceptions.recommendation import RecommendationUnavailableError
from port.movie_metadata import MetadataPage
from port.recommendation import RankedMovie, RankedRecommendations
from tests.factories import metadata


class Clock:
    def now(self):
        return datetime(2026, 1, 1, tzinfo=UTC)


class Seeds:
    def __init__(self):
        self.value = 0

    def new(self):
        self.value += 1
        return self.value


class Engine:
    async def status(self):
        return "ready"

    async def explore(self, ratings, limit, seed):
        return RankedRecommendations(
            tuple(
                RankedMovie(i, str(i), False, 0 if i > 30 else 0.5, exploratory=i > 30) for i in range(1, 61)
            ),
            "item_knn",
            "fixture",
            len(ratings),
            None,
        )


class Content:
    async def explore(self, preferences, limit, seed, year):
        return ContentCandidates(())


async def test_surprise_balances_candidates_excludes_rated_and_is_seeded():
    service = SurpriseRecommender(Engine(), Content(), RecommendationPolicy(), Clock(), Seeds())
    preferences = (MoviePreference(metadata(tmdb_id=1), 5),)
    first = await service.recommend(preferences, 10, 0)
    assert first == await service.recommend(preferences, 10, 0)
    assert first.items != (await service.recommend(preferences, 10, 1)).items
    assert len(first.items) == len({item.tmdb_id for item in first.items}) == 10
    assert 1 not in {item.tmdb_id for item in first.items}
    assert sum(item.exploratory for item in first.items) == 3
    assert first.mode == "surprise" and first.seed == 0
    assert (await service.recommend(preferences, 10)).seed == 1
    assert (await service.recommend(preferences, 10)).seed == 2


async def test_surprise_offline_model_fallback_and_total_failure():
    class Unavailable(Content):
        async def explore(self, *args):
            raise MovieMetadataUnavailableError

    service = SurpriseRecommender(Engine(), Unavailable(), RecommendationPolicy(), Clock(), Seeds())
    result = await service.recommend((), 10, 1)
    assert len(result.items) == 10 and result.degraded

    class NoModel(Engine):
        async def status(self):
            return "not_configured"

    with pytest.raises(RecommendationUnavailableError):
        await SurpriseRecommender(
            NoModel(), Unavailable(), RecommendationPolicy(), Clock(), Seeds()
        ).recommend((), 10)


async def test_metadata_explores_deep_year_pages_and_keeps_rare_movies():
    class Source:
        def __init__(self):
            self.calls = []

        async def get_exploration_page(self, **kwargs):
            self.calls.append(kwargs)
            return MetadataPage(
                (metadata(tmdb_id=999, popularity=0, vote_count=0), metadata()), kwargs["page"], 900, 18000
            )

    source = Source()
    recommender = MetadataContentRecommender(source)
    preferences = (MoviePreference(metadata(), 5),)
    result = await recommender.explore(preferences, 40, 42, 2026)
    assert [item.tmdb_id for item in result.items] == [999]
    assert not result.items[0].exploratory
    assert len(source.calls) <= 8
    assert any(call["page"] > 1 for call in source.calls)
    assert all(1 <= call["page"] <= 500 and 1900 <= call["year"] <= 2026 for call in source.calls)
    assert any(call["genre_id"] is None for call in source.calls)
    assert any(call["genre_id"] is not None for call in source.calls)
    again = await recommender.explore(preferences, 40, 42, 2026)
    assert again == result


async def test_metadata_rejects_negative_affinity_and_handles_partial_source():
    class Source:
        async def get_exploration_page(self, **kwargs):
            if kwargs["genre_id"] is not None:
                raise MovieMetadataUnavailableError
            return MetadataPage((metadata(tmdb_id=999),), 1, 1, 1)

    preferences = (MoviePreference(metadata(), 1), MoviePreference(metadata(tmdb_id=42, genres=()), 5))
    result = await MetadataContentRecommender(Source()).explore(preferences, 40, 1, 2026)
    assert not result.items
