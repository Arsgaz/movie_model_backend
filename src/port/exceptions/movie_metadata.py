class MovieMetadataError(Exception):
    """A metadata provider could not fulfill the request."""


class MovieMetadataUnavailableError(MovieMetadataError):
    pass


class MovieMetadataNotConfiguredError(MovieMetadataError):
    pass


class MovieMetadataNotFoundError(MovieMetadataError):
    pass
