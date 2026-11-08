from application.base_interactor import Interactor
from application.dto.catalog import ImportMovieCommand
from application.dto.movie import MovieDTO
from application.service.catalog_writer import CatalogWriter
from port.movie_metadata import MovieMetadataProvider


class ImportMovieHandler(Interactor[ImportMovieCommand, MovieDTO]):
    def __init__(self, metadata: MovieMetadataProvider, writer: CatalogWriter) -> None:
        self._metadata = metadata
        self._writer = writer

    async def __call__(self, data: ImportMovieCommand) -> MovieDTO:
        movie = await self._metadata.get_movie(media_type=data.media_type, tmdb_id=data.tmdb_id)
        return (await self._writer.save((movie,)))[0]
