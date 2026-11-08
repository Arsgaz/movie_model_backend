from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Genre:
    """TMDB genre identity is scoped by the parent movie's media type."""

    tmdb_id: int
    name: str
