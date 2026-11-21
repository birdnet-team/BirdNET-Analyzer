"""The input problems an evaluation reports with a code, for the GUI to translate."""


class EvaluationInputError(ValueError):
    """An input problem the GUI explains in the user's language.

    ``code`` names the problem and ``details`` holds the values the explanation
    needs; ``str()`` is the English message for the CLI and the logs.
    """

    def __init__(self, code: str, message: str, /, **details):
        super().__init__(message)
        self.code = code
        self.details = details
