import httpx2
from pydantic import ValidationError

from adapter.cache.ttl import AsyncTTLCache
from adapter.tmdb.dto import TMDBGenreListDTO, TMDBMovieDTO, TMDBPageDTO
from config.settings import Settings
from domain.entities.genre import Genre
from domain.value_objects.media_type import MediaType
from port.exceptions.movie_metadata import (
    MovieMetadataNotConfiguredError,
    MovieMetadataNotFoundError,
    MovieMetadataUnavailableError,
)
from port.movie_metadata import MetadataPage, MovieMetadata


class TMDBMovieMetadataProvider:
    def __init__(self, client: httpx2.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._key = settings.tmdb_api_key
        self._genres: AsyncTTLCache[MediaType, dict[int, str]] = AsyncTTLCache(max_entries=2, max_pending=2)

    async def _request(self, path: str, params: dict[str, str | int] | None = None) -> object:
        if self._key is None or not self._key.get_secret_value().strip():
            raise MovieMetadataNotConfiguredError
        try:
            response = await self._client.get(
                path,
                params={
                    "api_key": self._key.get_secret_value(),
                    "language": "ru-RU",
                    **(params or {}),
                },
            )
            if response.status_code == 404:
                raise MovieMetadataNotFoundError
            if not response.is_success:
                raise MovieMetadataUnavailableError
            return response.json()
        except (httpx2.HTTPError, ValueError):
            # Do not propagate HTTP errors or request URLs containing the credential.
            raise MovieMetadataUnavailableError from None

    @staticmethod
    def _normalize(item: TMDBMovieDTO, media_type: MediaType, genres: dict[int, str]) -> MovieMetadata:
        title = item.title if media_type == MediaType.MOVIE else item.name
        if not title or not title.strip() or len(title) > 500:
            raise MovieMetadataUnavailableError
        original = item.original_title if media_type == MediaType.MOVIE else item.original_name
        ids = item.genre_ids if item.genre_ids is not None else [genre.id for genre in (item.genres or [])]
        if item.genre_ids is None and item.genres is None:
            raise MovieMetadataUnavailableError
        if any(genre_id not in genres for genre_id in ids):
            raise MovieMetadataUnavailableError
        return MovieMetadata(
            tmdb_id=item.id,
            media_type=media_type,
            title=title.strip(),
            original_title=original or "",
            overview=item.overview or "",
            poster_path=item.poster_path,
            backdrop_path=item.backdrop_path,
            release_date=item.release_date if media_type == MediaType.MOVIE else item.first_air_date,
            vote_average=item.vote_average,
            vote_count=item.vote_count,
            popularity=item.popularity,
            genres=tuple(Genre(tmdb_id=genre_id, name=genres[genre_id]) for genre_id in sorted(set(ids))),
        )

    async def get_page(
        self,
        *,
        media_type: MediaType,
        query: str,
        genre_id: int | None,
        page: int,
    ) -> MetadataPage:
        params: dict[str, str | int] = {"page": page, "include_adult": "false"}
        if query:
            path = f"/search/{media_type.value}"
            params["query"] = query
        elif genre_id is not None:
            path = f"/discover/{media_type.value}"
            params.update(with_genres=genre_id, sort_by="popularity.desc")
        else:
            path = f"/{media_type.value}/popular"
        try:
            data = TMDBPageDTO.model_validate(await self._request(path, params))
            if data.page != page:
                raise MovieMetadataUnavailableError
            genres = {}
            if data.results:
                genres = await self._genres.get_or_load(
                    media_type,
                    lambda: self._get_genres(media_type),
                    ttl=86400,
                )
            items = tuple(self._normalize(item, media_type, genres) for item in data.results)
            # Search has no with_genres filter; counts describe the original TMDB page set.
            if genre_id is not None:
                items = tuple(item for item in items if any(g.tmdb_id == genre_id for g in item.genres))
            return MetadataPage(
                items=items, page=page, total_results=data.total_results, total_pages=data.total_pages
            )
        except ValidationError:
            raise MovieMetadataUnavailableError from None

    async def get_exploration_page(self, *, genre_id: int | None, year: int, page: int) -> MetadataPage:
        if not 1850 <= year <= 2100 or not 1 <= page <= 500 or (genre_id is not None and genre_id < 1):
            raise ValueError("Invalid exploration filters")
        params = {
            "page": page,
            "include_adult": "false",
            "include_video": "false",
            "primary_release_year": year,
            "sort_by": "popularity.desc",
        }
        if genre_id is not None:
            params["with_genres"] = genre_id
        try:
            data = TMDBPageDTO.model_validate(await self._request("/discover/movie", params))
            if data.page != page:
                raise MovieMetadataUnavailableError
            genres = (
                await self._genres.get_or_load(
                    MediaType.MOVIE, lambda: self._get_genres(MediaType.MOVIE), ttl=86400
                )
                if data.results
                else {}
            )
            return MetadataPage(
                tuple(self._normalize(item, MediaType.MOVIE, genres) for item in data.results),
                page,
                min(data.total_pages, 500),
                data.total_results,
            )
        except ValidationError:
            raise MovieMetadataUnavailableError from None

    async def get_movie(self, *, media_type: MediaType, tmdb_id: int) -> MovieMetadata:
        try:
            item = TMDBMovieDTO.model_validate(await self._request(f"/{media_type.value}/{tmdb_id}"))
            if item.id != tmdb_id or item.genres is None:
                raise MovieMetadataUnavailableError
            return self._normalize(item, media_type, {genre.id: genre.name for genre in item.genres})
        except ValidationError:
            raise MovieMetadataUnavailableError from None

    async def _get_genres(self, media_type: MediaType) -> dict[int, str]:
        data = TMDBGenreListDTO.model_validate(await self._request(f"/genre/{media_type.value}/list"))
        return {genre.id: genre.name for genre in data.genres}

    async def close(self) -> None:
        await self._genres.close()
