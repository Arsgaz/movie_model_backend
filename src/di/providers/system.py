from dishka import Provider, Scope, provide

from adapter.system.clock import SystemClock
from adapter.system.id_generator import UUIDGenerator
from adapter.system.random_seed import SecureRandomSeedGenerator
from port.clock import Clock
from port.id_generator import IdGenerator
from port.random_seed import RandomSeedGenerator


class SystemProvider(Provider):
    scope = Scope.APP

    @provide
    def clock(self) -> Clock:
        return SystemClock()

    @provide
    def ids(self) -> IdGenerator:
        return UUIDGenerator()

    @provide
    def random_seed(self) -> RandomSeedGenerator:
        return SecureRandomSeedGenerator()
