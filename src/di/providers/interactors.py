from dishka import Provider, Scope, provide

from application.command.change_library import ChangeLibraryHandler
from application.command.create_profile import CreateProfileHandler
from application.command.import_catalog import ImportCatalogHandler
from application.command.import_library import ImportLibraryHandler
from application.command.import_movie import ImportMovieHandler
from application.query.check_readiness import CheckReadinessHandler
from application.query.discover_movies import DiscoverMovieHandler, DiscoverMoviesHandler
from application.query.get_library import GetLibraryHandler
from application.query.get_movie import GetMovieHandler
from application.query.get_movie_by_tmdb import GetMovieByTMDBHandler
from application.query.get_profile import GetProfileHandler
from application.query.get_recommendations import GetRecommendationsHandler
from application.query.list_movies import ListMoviesHandler
from application.query.list_profiles import ListProfilesHandler
from application.service.catalog_writer import CatalogWriter
from application.service.movie_resolver import MovieResolver


class InteractorProvider(Provider):
    scope = Scope.REQUEST
    create_profile = provide(CreateProfileHandler)
    get_profile = provide(GetProfileHandler)
    list_profiles = provide(ListProfilesHandler)
    readiness = provide(CheckReadinessHandler)

    catalog_writer = provide(CatalogWriter)
    import_catalog = provide(ImportCatalogHandler)
    import_movie = provide(ImportMovieHandler)
    get_movie = provide(GetMovieHandler)
    list_movies = provide(ListMoviesHandler)

    get_movie_by_tmdb = provide(GetMovieByTMDBHandler)

    discover_movies = provide(DiscoverMoviesHandler)
    discover_movie = provide(DiscoverMovieHandler)

    movie_resolver = provide(MovieResolver)
    change_library = provide(ChangeLibraryHandler)
    get_library = provide(GetLibraryHandler)
    get_recommendations = provide(GetRecommendationsHandler)
    import_library = provide(ImportLibraryHandler)
