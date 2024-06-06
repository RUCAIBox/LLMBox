import inspect
from functools import wraps
from traceback import format_exc

from .logging import getFileLogger

ERROR_OVERVIEW = {
    "probability tensor contains either `inf`, `nan` or element < 0":
    "probability tensor contains either `inf`, `nan` or element < 0.\nSee https://github.com/meta-llama/llama/issues/380 for more details.",
    "trust_remote_code":
    "Unsupported datasets library version. Please update the datasets library to the latest version.\n\n  pip install datasets --upgrade",
    "'utf-8' codec can't decode byte 0x8b in position 1: invalid start byte":
    "Failed to fetch subset names from Hugging Face Hub. Please check your internet connection or try hf-mirror mode with `--hf_mirror` (experimental).",
}


def catch_error(continue_from_or_func: bool = False):
    """Catch the error and log the error message to log file. If the error is known, raise a RuntimeError with a message.

    Args:
        - continue_from_or_func (bool): Prompt the user to continue from the checkpoint if an error occurs.
    """

    def catch_error_decrator(func, continue_from=continue_from_or_func):
        """Catch the error and log the error message to log file."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (Exception, KeyboardInterrupt) as e:
                file_logger = getFileLogger()
                if continue_from and file_logger.handlers and hasattr(
                    file_logger.handlers[0], "evaluation_results_path"
                ):
                    from logging import getLogger

                    ckpt = file_logger.handlers[0].evaluation_results_path

                    logger = getLogger(__name__)
                    logger.warning(
                        f"Error occurred during evaluation. You can continue evaluation by loading the checkpoint: --continue_from {ckpt}"
                    )

                file_logger.error(f"[{func.__name__}] {e.__class__.__name__}: {e}\n\n{format_exc()}")
                for error, msg in ERROR_OVERVIEW.items():
                    if error in str(e):
                        raise RuntimeError(e.__class__.__name__ + ": " + msg) from e
                raise e

        return wrapper

    if inspect.isfunction(continue_from_or_func):
        return catch_error_decrator(continue_from_or_func, False)

    return catch_error_decrator
