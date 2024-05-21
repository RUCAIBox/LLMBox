from traceback import format_exc

from .logging import getFileLogger

ERROR_OVERVIEW = {
    "probability tensor contains either `inf`, `nan` or element < 0":
    "probability tensor contains either `inf`, `nan` or element < 0.\nSee https://github.com/meta-llama/llama/issues/380 for more details.",
    "trust_remote_code":
    "Unsupported datasets library version. Please update the datasets library to the latest version.\n\n  pip install datasets --upgrade",
}


def catch_error(func):
    """Catch the error and log the error message to log file."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (Exception, KeyboardInterrupt) as e:
            file_logger = getFileLogger()
            if file_logger.handlers and hasattr(file_logger.handlers[0], "evaluation_results_path"):
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
