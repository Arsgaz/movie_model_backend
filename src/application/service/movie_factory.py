from datetime import datetime
from uuid import UUID

from domain.entities.movie import Movie
from port.movie_metadata import MovieMetadata


def movie_from_metadata(item: MovieMetadata, movie_id: UUID, now: datetime) -> Movie:
    return Movie(
        id=movie_id,
        tmdb_id=item.tmdb_id,
        media_type=item.media_type,
        title=item.title,
        original_title=item.original_title,
        overview=item.overview,
        poster_path=item.poster_path,
        backdrop_path=item.backdrop_path,
        release_date=item.release_date,
        vote_average=item.vote_average,
        vote_count=item.vote_count,
        popularity=item.popularity,
        genres=item.genres,
        updated_at=now,
    )
