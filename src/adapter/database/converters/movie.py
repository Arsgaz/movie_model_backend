from datetime import UTC

from adapter.database.models.movie import MovieModel
from domain.entities.genre import Genre
from domain.entities.movie import Movie
from domain.value_objects.media_type import MediaType


def to_entity(model: MovieModel) -> Movie:
    timestamp = model.updated_at
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return Movie(
        id=model.id,
        tmdb_id=model.tmdb_id,
        media_type=MediaType(model.media_type),
        movielens_id=model.movielens_id,
        title=model.title,
        original_title=model.original_title,
        overview=model.overview,
        poster_path=model.poster_path,
        backdrop_path=model.backdrop_path,
        release_date=model.release_date,
        vote_average=model.vote_average,
        vote_count=model.vote_count,
        popularity=model.popularity,
        updated_at=timestamp,
        genres=tuple(Genre(tmdb_id=genre.tmdb_id, name=genre.name) for genre in model.genres),
    )
