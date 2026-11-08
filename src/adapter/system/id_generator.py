from uuid import UUID, uuid4


class UUIDGenerator:
    def new(self) -> UUID:
        return uuid4()
