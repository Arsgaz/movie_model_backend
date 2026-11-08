from application.base_interactor import Interactor
from application.dto.profile import ListProfilesQuery, ProfileDTO, ProfileListDTO
from port.repositories.profiles import ProfileRepository


class ListProfilesHandler(Interactor[ListProfilesQuery, ProfileListDTO]):
    def __init__(self, profiles: ProfileRepository) -> None:
        self._profiles = profiles

    async def __call__(self, data: ListProfilesQuery) -> ProfileListDTO:
        profiles = await self._profiles.list(limit=data.limit, offset=data.offset)
        return ProfileListDTO(
            items=tuple(ProfileDTO(id=p.id, name=p.name, created_at=p.created_at) for p in profiles),
            limit=data.limit,
            offset=data.offset,
        )
