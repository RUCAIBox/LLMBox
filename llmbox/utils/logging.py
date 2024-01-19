import datetime
import logging
from copy import copy
from os.path import abspath
from typing import Optional

import coloredlogs

DEFAULT_LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'

DEFAULT_DATETIME_FORMAT = '%Y_%m_%d-%H_%M_%S'  # Compatible with windows, which does not support ':' in filename

logger = logging.getLogger(__name__)

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def _get_file_handler(log_path, int_file_log_level):
    handler = logging.FileHandler(log_path)
    formatter = coloredlogs.BasicFormatter(fmt=DEFAULT_LOG_FORMAT)
    coloredlogs.HostNameFilter.install(handler=handler)
    handler.setLevel(level=int_file_log_level)
    handler.setFormatter(formatter)
    return handler


def _format_path(model_args, dataset_args, evaluation_args):
    model_name = model_args.model_name_or_path.strip("/").split("/")[-1]
    dataset_name = dataset_args.dataset_name + (
        "_" + ",".join(dataset_args.subset_names) if dataset_args.subset_names else ""
    )
    num_shots = str(dataset_args.num_shots)
    execution_time = datetime.datetime.now().strftime(DEFAULT_DATETIME_FORMAT)
    log_filename = f"{model_name}-{dataset_name}-{num_shots}shot-{execution_time}"
    log_path = f"{evaluation_args.logging_dir}/{log_filename}.log"
    evaluation_results_path = f"{evaluation_args.evaluation_results_dir}/{log_filename}.json"
    return log_path, evaluation_results_path


def getFileLogger(name=None):
    """Get a logger that only logs to file."""
    file_handler = None
    for hdlr in logging.getLogger(__package__.split(".")[0]).handlers:
        if isinstance(hdlr, logging.FileHandler):
            file_handler = hdlr
            break
    if file_handler is None:
        raise RuntimeError("The logging has not been initialized.")

    logger = copy(logging.getLogger(name))
    logger.handlers = [file_handler]
    logger.propagate = False
    return logger


def set_logging(
    model_args,
    dataset_args,
    evaluation_args,
    file_log_level: Optional[str] = None,
) -> None:
    """Set the logging level for standard output and file."""

    # Use package logger to disable logging of other packages. Set the level to DEBUG first
    # to allow all logs from our package, and then set the level to the desired one.
    package_logger = logging.getLogger(__package__.split(".")[0])
    if len(package_logger.handlers) != 0:
        raise RuntimeError("The logging has been initialized before.")

    # add stream handler to root logger
    coloredlogs.install(
        level=logging.DEBUG,
        logger=package_logger,
        fmt=DEFAULT_LOG_FORMAT,
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

    # finish logging initialization
    logger.info(f"Saving logs to {abspath(log_path)}")
