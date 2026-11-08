import secrets


class SecureRandomSeedGenerator:
    def new(self) -> int:
        return secrets.randbits(32)
