class UserFacingError(RuntimeError):
    """An error whose message is written for end users and is safe to show in the UI.
    Any other exception (e.g. RuntimeErrors raised by torch or other libraries) is shown
    only as a generic message; its details go to the server log."""
