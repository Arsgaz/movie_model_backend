from datetime import UTC

from adapter.database.models.profile import ProfileModel
from domain.entities.profile import Profile


def to_entity(model: ProfileModel) -> Profile:
    # SQLite returns a naive timestamp; storage always uses UTC.
    timestamp = model.created_at
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return Profile(id=model.id, name=model.name, created_at=timestamp)


def to_model(entity: Profile) -> ProfileModel:
    return ProfileModel(id=entity.id, name=entity.name, created_at=entity.created_at)
