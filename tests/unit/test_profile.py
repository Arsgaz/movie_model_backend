from datetime import UTC, datetime
from uuid import uuid4

import pytest

from domain.entities.profile import Profile
from domain.exceptions.profile import InvalidProfileNameError


@pytest.mark.parametrize("name", ["", "   ", "\t\n", "x" * 81])
def test_profile_rejects_invalid_name(name: str) -> None:
    with pytest.raises(InvalidProfileNameError):
        Profile(id=uuid4(), name=name, created_at=datetime.now(UTC))


def test_profile_trims_name_without_losing_unicode() -> None:
    profile = Profile(id=uuid4(), name="  Арсений  ", created_at=datetime.now(UTC))
    assert profile.name == "Арсений"
