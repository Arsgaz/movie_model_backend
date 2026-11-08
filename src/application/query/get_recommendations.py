import asyncio

from application.base_interactor import Interactor
from application.dto.recommendation import (
    GetRecommendationsQuery,
    RecommendationItemDTO,
    RecommendationsDTO,
)
from application.exceptions.profile import ProfileNotFoundError
from application.service.hybrid_recommendation import HybridRecommender, RecommendationPolicy
from application.service.surprise_recommendation import SurpriseRecommender
from domain.entities.genre import Genre
from domain.value_objects.media_type import MediaType
from port.content_recommendation import MoviePreference
from port.exceptions.movie_metadata import (
    MovieMetadataNotConfiguredError,
    MovieMetadataNotFoundError,
    MovieMetadataUnavailableError,
)
from port.movie_metadata import MovieMetadata, MovieMetadataProvider
from port.recommendation import RankedMovie
from port.repositories.library import LibraryRepository
from port.repositories.profiles import ProfileRepository
from port.unit_of_work import UnitOfWork


class GetRecommendationsHandler(Interactor[GetRecommendationsQuery, RecommendationsDTO]):
    def __init__(
        self,
        profiles: ProfileRepository,
        library: LibraryRepository,
        engine: HybridRecommender,
        metadata: MovieMetadataProvider,
        uow: UnitOfWork,
        policy: RecommendationPolicy,
        surprise: SurpriseRecommender,
    ) -> None:
        self._profiles, self._library, self._engine = profiles, library, engine
        self._metadata, self._uow = metadata, uow
        self._source_timeout = policy.source_timeout
        self._surprise = surprise

    async def __call__(self, data: GetRecommendationsQuery) -> RecommendationsDTO:
        maximum = 20 if data.mode == "surprise" else 50
        if not 1 <= data.limit <= maximum:
            raise ValueError(f"Limit must be between 1 and {maximum}")
        if await self._profiles.get(data.profile_id) is None:
            raise ProfileNotFoundError
        entries = await self._library.ratings(data.profile_id)
        preferences = tuple(
            MoviePreference(
                MovieMetadata(
                    tmdb_id=movie.tmdb_id,
                    media_type=movie.media_type,
                    title=movie.title,
                    original_title=movie.original_title,
                    overview=movie.overview,
                    poster_path=movie.poster_path,
                    backdrop_path=movie.backdrop_path,
                    release_date=movie.release_date,
                    vote_average=movie.vote_average,
                    vote_count=movie.vote_count,
                    popularity=movie.popularity,
                    genres=movie.genres,
                ),
                rating.value,
            )
            for rating, movie in entries
            if movie.media_type == MediaType.MOVIE
        )
        # Release the DB read transaction before inference and network candidate generation.
        await self._uow.rollback()
        if data.mode not in ("personalized", "surprise"):
            raise ValueError("Invalid recommendation mode")
        ranked = (
            await self._surprise.recommend(preferences, data.limit, data.seed)
            if data.mode == "surprise"
            else await self._engine.recommend(preferences, data.limit)
        )

        async def enrich(item: RankedMovie) -> tuple[RecommendationItemDTO, bool]:
            if item.metadata is not None:
                return RecommendationItemDTO(item.metadata, item.score, item.sources, item.exploratory), False
            try:
                async with asyncio.timeout(self._source_timeout):
                    movie = await self._metadata.get_movie(media_type=MediaType.MOVIE, tmdb_id=item.tmdb_id)
                return RecommendationItemDTO(movie, item.score, item.sources, item.exploratory), False
            except (
                MovieMetadataNotConfiguredError,
                MovieMetadataNotFoundError,
                MovieMetadataUnavailableError,
                TimeoutError,
            ):
                # Ranking remains available during TMDB outages; the detail page can retry later.
                movie = MovieMetadata(
                    tmdb_id=item.tmdb_id,
                    media_type=MediaType.MOVIE,
                    title=item.title,
                    original_title=item.title,
                    overview="",
                    poster_path=None,
                    backdrop_path=None,
                    release_date=None,
                    vote_average=0,
                    vote_count=0,
                    popularity=0,
                    genres=(Genre(16, "Анимация"),) if item.animation else (),
                )
                return RecommendationItemDTO(movie, item.score, item.sources, item.exploratory), True

        enriched = await asyncio.gather(*(enrich(item) for item in ranked.items))
        return RecommendationsDTO(
            items=tuple(item for item, _ in enriched),
            strategy=ranked.strategy,
            model_version=ranked.model_version,
            known_ratings=ranked.known_ratings,
            movie_ratings=len(preferences),
            fallback_reason=ranked.fallback_reason,
            metadata_partial=any(partial for _, partial in enriched),
            sources=ranked.sources,
            degraded=ranked.degraded,
            mode=ranked.mode,
            seed=ranked.seed,
        )
