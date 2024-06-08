import inspect
from functools import wraps
from traceback import format_exc

from .logging import getFileLogger

UNSOPPORTED_LIBRARY = "Unsupported {lib} library version. Please update the {lib} library to the latest version.\n\n  pip install {lib} --upgrade"

ERROR_OVERVIEW = {
    "probability tensor contains either `inf`, `nan` or element < 0":
    "probability tensor contains either `inf`, `nan` or element < 0.\nSee https://github.com/meta-llama/llama/issues/380 for more details.",
    "'utf-8' codec can't decode byte 0x8b in position 1: invalid start byte":
    "Failed to fetch subset names from Hugging Face Hub. Please check your internet connection or try hf-mirror mode with `--hf_mirror` (experimental).",
    "openai.types":
    UNSOPPORTED_LIBRARY.format(lib="openai"),
    "trust_remote_code":
    UNSOPPORTED_LIBRARY.format(lib="datasets"),
    "datasets.exceptions.DatasetGenerationError":
    "There is some issue when loading dataset with threading. Please try to disable threading with `--no_dataset_threading`.",
    "assert logits_applied == logits.shape[0]":
    "Current version of vLLM does not support prefix_caching. Try disable vLLM or disable prefix_caching.",
}


def catch_error(continue_from: bool = False):
    """Catch the error and log the error message to log file. If the error is known, raise a RuntimeError with a message.

    Args:
        - continue_from (bool): Prompt the user to continue from the checkpoint if an error occurs.
    """

    def catch_error_decrator(func):
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

    return catch_error_decrator
