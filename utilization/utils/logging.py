import datetime
import logging
import os
import pathlib
from typing import TYPE_CHECKING, Optional

import coloredlogs

if TYPE_CHECKING:
    # solve the circular import
    from .arguments import DatasetArguments, EvaluationArguments, ModelArguments

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
DEBUG_LOG_FORMAT = "%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s"

DEFAULT_DATETIME_FORMAT = "%Y_%m_%d-%H_%M_%S"  # Compatible with windows, which does not support ':' in filename

logger = logging.getLogger(__name__)

llmbox_package = __package__.split(".")[0]

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def get_git_revision(base_path) -> str:
    try:
        git_dir = pathlib.Path(base_path) / '.git'
        with (git_dir / 'HEAD').open('r') as head:
            ref = head.readline()

        if ref.startswith("ref:"):
            ref = ref.split(' ')[-1].strip()
            with (git_dir / ref).open('r') as git_hash:
                return git_hash.readline().strip()
        else:
            return ref.strip()
    except FileNotFoundError:
        return "Not a git repository"


def _get_file_handler(log_path, int_file_log_level):
    handler = logging.FileHandler(log_path)
    formatter = coloredlogs.BasicFormatter(
        fmt=DEFAULT_LOG_FORMAT if int_file_log_level != logging.DEBUG else DEBUG_LOG_FORMAT
    )
    coloredlogs.HostNameFilter.install(handler=handler)
    handler.setLevel(level=int_file_log_level)
    handler.setFormatter(formatter)
    return handler


def _format_path(model_args, dataset_args, evaluation_args):
    model_name = model_args.model_name_or_path.strip("/").split("/")[-1]
    dataset_name = dataset_args.dataset_name.replace(
        '/', '-'
    ) + ("_" + ",".join(dataset_args.subset_names) if dataset_args.subset_names else "")
    num_shots = str(dataset_args.num_shots)
    execution_time = datetime.datetime.now().strftime(DEFAULT_DATETIME_FORMAT)
    log_filename = f"{model_name}-{dataset_name}-{num_shots}shot-{execution_time}"
    log_path = f"{evaluation_args.logging_dir}/{log_filename}.log"
    evaluation_results_path = f"{evaluation_args.evaluation_results_dir}/{log_filename}.json"
    return log_path, evaluation_results_path


def getFileLogger(name=None):
    """Get a logger that only logs to file."""
    return logging.getLogger("file_" + (name or llmbox_package))


def set_logging(
    model_args: "ModelArguments",
    dataset_args: "DatasetArguments",
    evaluation_args: "EvaluationArguments",
    file_log_level: Optional[str] = None,
) -> None:
    """Set the logging level for standard output and file."""

    # Use package logger to disable logging of other packages. Set the level to DEBUG first
    # to allow all logs from our package, and then set the level to the desired one.
    package_logger = logging.getLogger(llmbox_package)
    if len(package_logger.handlers) != 0:
        return

    # add stream handler to root logger
    coloredlogs.install(
        level=logging.DEBUG,
        logger=package_logger,
        fmt=DEFAULT_LOG_FORMAT if evaluation_args.log_level != "debug" else DEBUG_LOG_FORMAT,
    )
    package_logger.handlers[0].setLevel(level=log_levels[evaluation_args.log_level])

    # set the log file
    log_path, evaluation_results_path = _format_path(model_args, dataset_args, evaluation_args)
    dataset_args.evaluation_results_path = evaluation_results_path  # type: ignore

    # add file handler to root logger
    if file_log_level is None:
        int_file_log_level = min(log_levels[evaluation_args.log_level], log_levels["info"])
    else:
        int_file_log_level = log_levels[file_log_level]
    handler = _get_file_handler(log_path, int_file_log_level)
    package_logger.addHandler(handler)

    file_package_logger = logging.getLogger("file_" + llmbox_package)
    file_package_logger.setLevel(package_logger.level)
    file_package_logger.handlers = [handler]
    file_package_logger.propagate = False

    # finish logging initialization
    logger.info(f"Saving logs to {os.path.abspath(log_path)}")
    getFileLogger().info(f"LLMBox revision: {get_git_revision(os.path.join(os.path.dirname(__file__), '../..'))}")
