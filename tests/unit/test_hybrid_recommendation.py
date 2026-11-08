import asyncio

import pytest

from application.service.content_recommendation import MetadataContentRecommender
from application.service.hybrid_recommendation import HybridRecommender, RecommendationPolicy
from domain.entities.genre import Genre
from port.content_recommendation import ContentCandidates, MoviePreference
from port.exceptions.movie_metadata import MovieMetadataUnavailableError
from port.exceptions.recommendation import RecommendationUnavailableError
from port.movie_metadata import MetadataPage
from port.recommendation import RankedMovie, RankedRecommendations
from tests.factories import FakeMetadata, metadata


def candidate(identity, score=1):
    return RankedMovie(identity, str(identity), False, score)


class Engine:
    def __init__(self, items=(), *, status="ready", strategy="item_knn"):
        self.items, self.state, self.strategy = items, status, strategy
        self.calls = []

    async def status(self):
        return self.state

    async def recommend(self, ratings, limit):
        self.calls.append((dict(ratings), limit))
        return RankedRecommendations(self.items, self.strategy, "fixture", len(ratings), None)


class Content:
    def __init__(self, items=(), popular=(), *, error=None):
        self.items, self.tail, self.error = items, popular, error
        self.calls = []

    async def recommend(self, preferences, limit):
        self.calls.append((preferences, limit))
        if self.error:
            raise self.error
        return ContentCandidates(self.items)

    async def popular(self, excluded, limit):
        if self.error:
            raise self.error
        return ContentCandidates(self.tail)


@pytest.mark.asyncio
async def test_hybrid_fuses_ranks_not_raw_scores_and_deduplicates_sources():
    engine = Engine((candidate(1, 100000), candidate(2, 999), candidate(2, 900)))
    content = Content((candidate(3, 0.01), candidate(2, 0.0001)))
    service = HybridRecommender(engine, content, RecommendationPolicy())
    result = await service.recommend((), 3)
    assert [item.tmdb_id for item in result.items] == [2, 1, 3]
    assert result.items[0].sources == ("collaborative", "content")
    assert result.items[0].score == pytest.approx(2 / 62)
    assert result.strategy == "hybrid" and not result.degraded


@pytest.mark.asyncio
async def test_rated_movies_are_excluded_from_every_source_and_fallback():
    preference = MoviePreference(metadata(tmdb_id=1), 1)
    service = HybridRecommender(
        Engine((candidate(1), candidate(2))),
        Content((candidate(1), candidate(3)), (candidate(1), candidate(2), candidate(4))),
        RecommendationPolicy(),
    )
    result = await service.recommend((preference,), 4)
    assert [item.tmdb_id for item in result.items] == [2, 3, 4]
    assert result.items[-1].sources == ("popularity",) and result.items[-1].score == 0


@pytest.mark.parametrize("state", ["not_configured", "unavailable"])
@pytest.mark.asyncio
async def test_content_supports_movies_outside_model_and_missing_artifact(state):
    service = HybridRecommender(Engine(status=state), Content((candidate(999999),)), RecommendationPolicy())
    result = await service.recommend((MoviePreference(metadata(), 5),), 1)
    assert result.strategy == "content" and result.model_version is None
    assert result.items[0].tmdb_id == 999999
    assert result.known_ratings == 0
    assert result.degraded is (state == "unavailable")


@pytest.mark.asyncio
async def test_metadata_outage_keeps_collaborative_results():
    service = HybridRecommender(
        Engine((candidate(1),)), Content(error=MovieMetadataUnavailableError()), RecommendationPolicy()
    )
    result = await service.recommend((), 1)
    assert result.strategy == "item_knn" and result.degraded
    assert result.items[0].sources == ("collaborative",)


@pytest.mark.asyncio
async def test_empty_profile_without_model_uses_popularity():
    service = HybridRecommender(
        Engine(status="not_configured"), Content(popular=(candidate(1),)), RecommendationPolicy()
    )
    result = await service.recommend((), 2)
    assert result.strategy == "popularity" and result.fallback_reason == "model_not_configured"
    assert result.items[0].sources == ("popularity",)


@pytest.mark.asyncio
async def test_no_working_sources_is_an_error_not_empty_success():
    service = HybridRecommender(
        Engine(status="not_configured"),
        Content(error=MovieMetadataUnavailableError()),
        RecommendationPolicy(),
    )
    with pytest.raises(RecommendationUnavailableError):
        await service.recommend((), 10)


@pytest.mark.asyncio
async def test_source_timeout_does_not_block_other_sources():
    class SlowContent(Content):
        async def recommend(self, preferences, limit):
            await asyncio.Event().wait()

    service = HybridRecommender(
        Engine((candidate(1),)), SlowContent(), RecommendationPolicy(source_timeout=0.01)
    )
    result = await asyncio.wait_for(service.recommend((), 1), timeout=0.5)
    assert result.strategy == "item_knn" and result.degraded


@pytest.mark.asyncio
async def test_content_uses_genres_and_text_with_two_pages_max_and_does_not_import():
    class Pages(FakeMetadata):
        async def get_page(self, **kwargs):
            self.calls.append(kwargs)
            return MetadataPage(
                (
                    metadata(tmdb_id=123),
                    metadata(tmdb_id=1000000, overview="Космический корабль путешествует среди планет"),
                    metadata(tmdb_id=1000001, genres=(Genre(99, "Документальный"),), overview="Другой сюжет"),
                ),
                1,
                1,
                3,
            )

    source = Pages()
    content = MetadataContentRecommender(source)
    seeds = (MoviePreference(metadata(overview="Космический корабль путешествует среди планет"), 5),)
    result = await content.recommend(seeds, 20)
    assert [item.tmdb_id for item in result.items] == [1000000]
    assert result.items[0].metadata is not None
    assert len(source.calls) == 2 and all(call["page"] == 1 for call in source.calls)
    negative = MoviePreference(metadata(tmdb_id=456, overview=seeds[0].movie.overview), 1)
    result = await content.recommend(seeds + (negative,), 20)
    assert not result.items


@pytest.mark.asyncio
async def test_one_failed_discovery_page_marks_content_partial():
    class Pages(FakeMetadata):
        async def get_page(self, **kwargs):
            if kwargs["genre_id"] == 16:
                raise MovieMetadataUnavailableError
            return MetadataPage((metadata(tmdb_id=900),), 1, 1, 1)

    result = await MetadataContentRecommender(Pages()).recommend((MoviePreference(metadata(), 5),), 10)
    assert result.partial and result.items[0].tmdb_id == 900


@pytest.mark.asyncio
async def test_negative_only_profiles_do_not_query_content():
    source = FakeMetadata()
    result = await MetadataContentRecommender(source).recommend((MoviePreference(metadata(), 1),), 10)
    assert not result.items and not source.calls


async def test_large_selection_fills_fifty_unique_unrated_movies_from_expanded_pool():
    engine = Engine(tuple(candidate(i) for i in range(1, 45)))
    content = Content(tuple(candidate(i) for i in range(25, 61)))
    service = HybridRecommender(engine, content, RecommendationPolicy())
    result = await service.recommend((MoviePreference(metadata(tmdb_id=1), 5),), 50)
    assert len(result.items) == len({item.tmdb_id for item in result.items}) == 50
    assert 1 not in {item.tmdb_id for item in result.items}
    assert engine.calls[-1][1] == content.calls[-1][1] == 100


async def test_larger_content_pool_reads_second_genre_pages_and_third_popular_page():
    class Pages(FakeMetadata):
        async def get_page(self, **kwargs):
            self.calls.append(kwargs)
            identity = 1000 + (kwargs["genre_id"] or 0) * 10 + kwargs["page"]
            return MetadataPage((metadata(tmdb_id=identity),), kwargs["page"], 3, 60)

    source = Pages()
    recommender = MetadataContentRecommender(source)
    result = await recommender.recommend((MoviePreference(metadata(), 5),), 60)
    assert len(result.items) == 4
    assert len(source.calls) == 4 and {call["page"] for call in source.calls} == {1, 2}
    source.calls.clear()
    popular = await recommender.popular(frozenset(), 60)
    assert len(popular.items) == 3
    assert [call["page"] for call in source.calls] == [1, 2, 3]
