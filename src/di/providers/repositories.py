from dishka import Provider, Scope, provide
from sqlalchemy.ext.asyncio import AsyncSession

from adapter.database.repositories.library import SQLAlchemyLibraryRepository
from adapter.database.repositories.movies import SQLAlchemyMovieRepository
from adapter.database.repositories.profiles import SQLAlchemyProfileRepository
from port.repositories.library import LibraryRepository
from port.repositories.movies import MovieRepository
from port.repositories.profiles import ProfileRepository


class RepositoryProvider(Provider):
    scope = Scope.REQUEST

    @provide
    def profiles(self, session: AsyncSession) -> ProfileRepository:
        return SQLAlchemyProfileRepository(session)

    @provide
    def movies(self, session: AsyncSession) -> MovieRepository:
        return SQLAlchemyMovieRepository(session)

    @provide
    def library(self, session: AsyncSession) -> LibraryRepository:
        return SQLAlchemyLibraryRepository(session)
