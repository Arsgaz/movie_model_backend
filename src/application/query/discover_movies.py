from application.base_interactor import Interactor
from application.dto.discovery import DiscoverMovieQuery, DiscoverMoviesQuery, DiscoveryPageDTO
from port.movie_metadata import MovieMetadata, MovieMetadataProvider


class DiscoverMoviesHandler(Interactor[DiscoverMoviesQuery, DiscoveryPageDTO]):
    def __init__(self, metadata: MovieMetadataProvider) -> None:
        self._metadata = metadata

    async def __call__(self, data: DiscoverMoviesQuery) -> DiscoveryPageDTO:
        query = data.query.strip()
        page = await self._metadata.get_page(
            media_type=data.media_type,
            query=query,
            genre_id=data.genre_id,
            page=data.page,
        )
        return DiscoveryPageDTO(
            items=page.items,
            page=page.page,
            total_pages=min(page.total_pages, 500),
            # TMDB search cannot filter genres; this filter applies to each source page.
            total_results=None if query and data.genre_id is not None else page.total_results,
        )


class DiscoverMovieHandler(Interactor[DiscoverMovieQuery, MovieMetadata]):
    def __init__(self, metadata: MovieMetadataProvider) -> None:
        self._metadata = metadata

    async def __call__(self, data: DiscoverMovieQuery) -> MovieMetadata:
        return await self._metadata.get_movie(media_type=data.media_type, tmdb_id=data.tmdb_id)
