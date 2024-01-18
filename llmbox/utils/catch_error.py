from traceback import format_exc

from .logging import getFileLogger


def catch_error(func):
    """Catch the error and log the error message to log file."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            file_logger = getFileLogger(func.__module__)
            file_logger.error(f"[{func.__name__}] {e.__class__.__name__}: {e}\n\n{format_exc()}")
            raise e
    return wrapper
