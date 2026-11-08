from application.base_interactor import Interactor
from application.dto.catalog import CatalogImportDTO, ImportCatalogPageCommand
from application.service.catalog_writer import CatalogWriter
from port.movie_metadata import MovieMetadataProvider


class ImportCatalogHandler(Interactor[ImportCatalogPageCommand, CatalogImportDTO]):
    def __init__(self, metadata: MovieMetadataProvider, writer: CatalogWriter) -> None:
        self._metadata = metadata
        self._writer = writer

    async def __call__(self, data: ImportCatalogPageCommand) -> CatalogImportDTO:
        # Finish remote I/O before beginning the write transaction.
        page = await self._metadata.get_page(
            media_type=data.media_type,
            query=data.query.strip(),
            genre_id=data.genre_id,
            page=data.page,
        )
        items = await self._writer.save(page.items)
        return CatalogImportDTO(
            items=items,
            page=page.page,
            imported_count=len(items),
            source_total_results=page.total_results,
            source_total_pages=page.total_pages,
        )
