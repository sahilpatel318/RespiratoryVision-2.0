class InputError(ValueError):
    """Malformed/unsupported recording, with a stable quality-state code."""

    def __init__(self, message: str, state: str = "UNSUPPORTED_INPUT"):
        super().__init__(message)
        self.state = state
