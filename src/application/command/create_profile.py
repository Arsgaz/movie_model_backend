from application.base_interactor import Interactor
from application.dto.profile import CreateProfileCommand, ProfileDTO
from domain.entities.profile import Profile
from port.clock import Clock
from port.id_generator import IdGenerator
from port.repositories.profiles import ProfileRepository
from port.unit_of_work import UnitOfWork


class CreateProfileHandler(Interactor[CreateProfileCommand, ProfileDTO]):
    def __init__(self, profiles: ProfileRepository, uow: UnitOfWork, clock: Clock, ids: IdGenerator) -> None:
        self._profiles = profiles
        self._uow = uow
        self._clock = clock
        self._ids = ids

    async def __call__(self, data: CreateProfileCommand) -> ProfileDTO:
        profile = Profile(id=self._ids.new(), name=data.name, created_at=self._clock.now())
        try:
            await self._profiles.add(profile)
            await self._uow.commit()
        except Exception:
            await self._uow.rollback()
            raise
        return ProfileDTO(id=profile.id, name=profile.name, created_at=profile.created_at)
