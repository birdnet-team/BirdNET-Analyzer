"""Errors of the training that the GUI explains in the user's language."""


class TrainingInputError(ValueError):
    """A problem with the training data or its settings.

    ``key`` names the GUI text and ``values`` fill its placeholders; ``str()`` is the
    English message for the CLI and the logs.
    """

    def __init__(self, key: str, message: str, /, *values: str):
        super().__init__(message)
        self.key = key
        self.values = values

    def __reduce__(self):
        return type(self), (self.key, str(self), *self.values)
