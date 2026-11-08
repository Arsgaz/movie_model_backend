from application.base_interactor import Interactor
from application.dto.profile import GetProfileQuery, ProfileDTO
from application.exceptions.profile import ProfileNotFoundError
from port.repositories.profiles import ProfileRepository


class GetProfileHandler(Interactor[GetProfileQuery, ProfileDTO]):
    def __init__(self, profiles: ProfileRepository) -> None:
        self._profiles = profiles

    async def __call__(self, data: GetProfileQuery) -> ProfileDTO:
        profile = await self._profiles.get(data.profile_id)
        if profile is None:
            raise ProfileNotFoundError
        return ProfileDTO(id=profile.id, name=profile.name, created_at=profile.created_at)
