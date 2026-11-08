import httpx2
import pytest

from adapter.tmdb.movie_metadata import TMDBMovieMetadataProvider
from config.settings import Settings
from domain.value_objects.media_type import MediaType
from port.exceptions.movie_metadata import (
    MovieMetadataNotConfiguredError,
    MovieMetadataNotFoundError,
    MovieMetadataUnavailableError,
)


def item(**changes: object) -> dict:
    return {
        "id": 123,
        "title": "Фильм",
        "original_title": "Movie",
        "genre_ids": [16],
        "release_date": "",
        "poster_path": None,
        "overview": None,
        **changes,
    }


def settings() -> Settings:
    return Settings(_env_file=None, tmdb_api_key="test-credential")


async def test_page_uses_tmdb_base_path_and_preserves_filtered_source_counts() -> None:
    requests = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        assert request.url.params["api_key"] == "test-credential"
        if request.url.path == "/3/genre/movie/list":
            return httpx2.Response(200, json={"genres": [{"id": 16, "name": "Анимация"}]})
        assert request.url.path == "/3/search/movie"
        assert request.url.params["query"] == "Фильм"
        assert "with_genres" not in request.url.params
        return httpx2.Response(
            200,
            json={
                "page": 2,
                "total_results": 100,
                "total_pages": 5,
                "results": [item(), item(id=124, genre_ids=[])],
            },
        )

    async with httpx2.AsyncClient(
        base_url="https://api.themoviedb.org/3/", transport=httpx2.MockTransport(respond)
    ) as client:
        provider = TMDBMovieMetadataProvider(client, settings())
        page = await provider.get_page(media_type=MediaType.MOVIE, query="Фильм", genre_id=16, page=2)
    assert len(page.items) == 1
    assert page.items[0].genres[0].name == "Анимация"
    assert page.items[0].release_date is None and page.items[0].overview == ""
    assert page.total_results == 100 and page.total_pages == 5
    assert len(requests) == 2


@pytest.mark.parametrize(
    "media_type,genre_id,expected_path",
    [
        (MediaType.MOVIE, None, "/3/movie/popular"),
        (MediaType.TV, 16, "/3/discover/tv"),
    ],
)
async def test_popular_discover_and_empty_page(
    media_type: MediaType, genre_id: int | None, expected_path: str
) -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == expected_path
        if genre_id:
            assert request.url.params["with_genres"] == str(genre_id)
        return httpx2.Response(200, json={"page": 1, "total_results": 0, "total_pages": 0, "results": []})

    async with httpx2.AsyncClient(
        base_url="https://api.themoviedb.org/3/", transport=httpx2.MockTransport(respond)
    ) as client:
        page = await TMDBMovieMetadataProvider(client, settings()).get_page(
            media_type=media_type,
            query="",
            genre_id=genre_id,
            page=1,
        )
        assert not page.items


async def test_tv_details_use_tv_identity_and_nullable_fields() -> None:
    data = {
        "id": 123,
        "name": "Сериал",
        "original_name": "Series",
        "first_air_date": "2020-01-01",
        "genres": [{"id": 18, "name": "Драма"}],
    }
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json=data)),
        base_url="https://api.themoviedb.org/3/",
    ) as client:
        movie = await TMDBMovieMetadataProvider(client, settings()).get_movie(
            media_type=MediaType.TV, tmdb_id=123
        )
        assert movie.title == "Сериал" and movie.original_title == "Series"
        assert movie.media_type == MediaType.TV and str(movie.release_date) == "2020-01-01"
        assert movie.poster_path is None


@pytest.mark.parametrize(
    "status,error",
    [
        (404, MovieMetadataNotFoundError),
        (401, MovieMetadataUnavailableError),
        (429, MovieMetadataUnavailableError),
        (500, MovieMetadataUnavailableError),
    ],
)
async def test_http_failures_are_translated(status: int, error: type[Exception]) -> None:
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(status)),
        base_url="https://api.themoviedb.org/3/",
    ) as client:
        with pytest.raises(error) as exc:
            await TMDBMovieMetadataProvider(client, settings()).get_movie(
                media_type=MediaType.MOVIE, tmdb_id=123
            )
        assert "test-credential" not in str(exc.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"id": 123, "title": "", "genres": []},
        {"id": 124, "title": "Unexpected identity", "genres": []},
        {"id": 123, "title": "Missing genres"},
        {"id": 123, "title": "Invalid date", "genres": [], "release_date": "invalid"},
        {"id": 123, "title": "Invalid rating", "genres": [], "vote_average": -1},
        {
            "id": 123,
            "title": "Bad poster",
            "genres": [],
            "poster_path": "https://unrelated.example/image.jpg",
        },
    ],
)
async def test_malformed_metadata_is_rejected(payload: dict) -> None:
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json=payload)),
        base_url="https://api.themoviedb.org/3/",
    ) as client:
        with pytest.raises(MovieMetadataUnavailableError):
            await TMDBMovieMetadataProvider(client, settings()).get_movie(
                media_type=MediaType.MOVIE, tmdb_id=123
            )


async def test_transport_errors_do_not_leak_request_credentials() -> None:
    def fail(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("Failed URL containing test-credential", request=request)

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(fail), base_url="https://api.themoviedb.org/3/"
    ) as client:
        with pytest.raises(MovieMetadataUnavailableError) as exc:
            await TMDBMovieMetadataProvider(client, settings()).get_movie(
                media_type=MediaType.MOVIE, tmdb_id=123
            )
        assert "test-credential" not in str(exc.value)
        assert exc.value.__suppress_context__


async def test_missing_key_never_sends_a_request() -> None:
    def fail(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("Network must not be used")

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(fail), base_url="https://api.themoviedb.org/3/"
    ) as client:
        provider = TMDBMovieMetadataProvider(client, Settings(_env_file=None, tmdb_api_key=None))
        with pytest.raises(MovieMetadataNotConfiguredError):
            await provider.get_movie(media_type=MediaType.MOVIE, tmdb_id=123)


async def test_exploration_uses_year_genre_and_deep_page_without_vote_threshold():
    def respond(request):
        if request.url.path.endswith("genre/movie/list"):
            return httpx2.Response(200, json={"genres": [{"id": 16, "name": "Анимация"}]})
        params = request.url.params
        assert request.url.path == "/3/discover/movie"
        assert params["primary_release_year"] == "1995" and params["page"] == "400"
        assert params["with_genres"] == "16" and params["include_adult"] == "false"
        assert "vote_count.gte" not in params
        return httpx2.Response(
            200,
            json={"page": 400, "total_results": 18000, "total_pages": 900, "results": [item(vote_count=0)]},
        )

    async with httpx2.AsyncClient(
        base_url="https://api.themoviedb.org/3/", transport=httpx2.MockTransport(respond)
    ) as client:
        source = TMDBMovieMetadataProvider(client, settings())
        result = await source.get_exploration_page(genre_id=16, year=1995, page=400)
        assert result.total_pages == 500 and result.items[0].vote_count == 0
        await source.close()
