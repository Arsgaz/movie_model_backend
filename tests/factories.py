from dataclasses import replace
from datetime import date

from domain.entities.genre import Genre
from domain.value_objects.media_type import MediaType
from port.movie_metadata import MetadataPage, MovieMetadata


def metadata(**changes: object) -> MovieMetadata:
    item = MovieMetadata(
        tmdb_id=123,
        media_type=MediaType.MOVIE,
        title="Тайна Коко",
        original_title="Coco",
        overview="Описание",
        poster_path="/poster.jpg",
        backdrop_path=None,
        release_date=date(2017, 10, 27),
        vote_average=8.2,
        vote_count=100,
        popularity=20.0,
        genres=(Genre(tmdb_id=16, name="Мультфильм"), Genre(tmdb_id=12, name="Приключения")),
    )
    return replace(item, **changes)


class FakeMetadata:
    def __init__(self) -> None:
        self.items = (metadata(),)
        self.error: Exception | None = None
        self.calls = []

    async def get_page(self, **kwargs: object) -> MetadataPage:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return MetadataPage(
            items=self.items, page=kwargs["page"], total_results=len(self.items), total_pages=1
        )

    async def get_exploration_page(self, **kwargs: object) -> MetadataPage:
        return await self.get_page(**kwargs)

    async def get_movie(self, **kwargs: object) -> MovieMetadata:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return metadata(media_type=kwargs["media_type"], tmdb_id=kwargs["tmdb_id"])
