import importlib
import inspect
import logging
from dataclasses import MISSING, dataclass
from logging import getLogger
from typing import Optional, Tuple, Type, TypeVar
import datetime
import warnings

import coloredlogs
from transformers.hf_argparser import HfArg, HfArgumentParser

T = TypeVar('T')

logger = getLogger(__name__)

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

DEFAULT_LOG_FORMAT = '%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s'

DEFAULT_DATETIME_FORMAT = '%Y_%m_%d-%H_%M_%S'       # Compatible with windows, which does not support ':' in filename


@dataclass
class ModelArguments:

    model_name_or_path: str = HfArg(
        default=MISSING, aliases=["--model", "-m"], help="The model name or path, e.g., cuire, llama"
    )
    openai_api_key: str = HfArg(
        default="",
        help="The OpenAI API key",
    )
    load_in_half: bool = HfArg(
        default=True,
        help="Whether to load the model in half precision",
    )
    device_map: str = HfArg(
        default="auto",
        help="The device map for model and data",
    )


@dataclass
class DatasetArguments:

    dataset: str = HfArg(default=MISSING, aliases=["-d"], help="The dataset name, e.g., copa, gsm")
    evaluation_set: str = HfArg(
        default="validation",
        help="The set name for evaluation, e.g., validation, test",
    )
    example_set: str = HfArg(
        default="train",
        help="The set name for demonstration, e.g., train, dev",
    )
    system_prompt: str = HfArg(
        aliases=["-sys"],
        default="",
        help="The system prompt of the model",
    )
    instance_format: str = HfArg(
        aliases=['-fmt'],
        default="{source}{target}",
        help="The format to format the `source` and `target` for each instance",
    )
    num_shots: int = HfArg(
        aliases=['-shots'],
        default=0,
        help="The few-shot number for demonstration",
    )
    max_example_tokens: int = HfArg(
        default=1024,
        help="The maximum token number of demonstration",
    )
    batch_size: int = HfArg(
        default=1,
        aliases=["-bsz"],
        help="The evaluation batch size",
    )
    trust_remote_code: bool = HfArg(
        default=False,
        help="Whether to trust the remote code",
    )


@dataclass
class EvaluationArguments:

    seed: int = HfArg(
        default=2023,
        help="The random seed",
    )
    logging_dir: str = HfArg(
        default="logs",
        help="The logging directory",
    )
    log_level: Optional[str] = HfArg(
        default="warning",
        help=
        "Logger level to use on the main node.ossible choices are the log levels as strings: 'debug', 'info', 'warning', 'error' and 'critical'",
        metadata={"choices": log_levels.keys()},
    )
    use_sc: bool = HfArg(
        default=False,
        help="Whether to use self-consistency",
    )
    sample_path: int = HfArg(
        default=1,
        help="The path number for sampling for self-consistency",
    )


def set_logging(
    model_args: ModelArguments,
    dataset_args: DatasetArguments,
    evaluation_args: EvaluationArguments,
    file_log_level: str = 'info',
) -> None:
    """Set the logging level for standard output and file."""

    # use root logger to disable logging of other packages
    root_logger = logging.getLogger(__package__)
    coloredlogs.install(
        level=log_levels[evaluation_args.log_level],
        logger=root_logger,
        fmt=DEFAULT_LOG_FORMAT,
    )

    # set the log file
    model_name = model_args.model_name_or_path.strip("/").split("/")[-1]
    dataset_name = dataset_args.dataset
    num_shots = str(dataset_args.num_shots)
    execution_time = datetime.datetime.now().strftime(DEFAULT_DATETIME_FORMAT)
    log_filename = f"{model_name}-{dataset_name}-{num_shots}-{execution_time}.log"
    log_path = f"{evaluation_args.logging_dir}/{log_filename}"

    # add file handler to root logger
    handler = logging.FileHandler(log_path)
    formatter = coloredlogs.BasicFormatter(fmt=DEFAULT_LOG_FORMAT)
    handler.setLevel(level=log_levels[file_log_level])
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # finish logging initialization
    logger.info(f"Saving logs to {log_path}")


def check_args(model_args, dataset_args, evaluation_args):
    r"""Check the validity of arguments.

    Args:
        model_args (ModelArguments): The global configurations.
        dataset_args (DatasetArguments): The dataset configurations.
        evaluation_args (EvaluationArguments): The evaluation configurations.
    """
    if model_args.model_name_or_path.lower() == 'gpt-3.5-turbo' and dataset_args.batch_size > 1:
        dataset_args.batch_size = 1
        warnings.warn("gpt-3.5-turbo doesn't support batch_size > 1, automatically set batch_size=1.")

def parse_argument() -> Tuple[ModelArguments, DatasetArguments, EvaluationArguments]:
    r"""Parse arguments from command line. Using `argparse` for predefined ones, and an easy mannal parser for others (saved in `kwargs`).

    Returns:
        Namespace: the parsed arguments
    """
    parser = HfArgumentParser((ModelArguments, DatasetArguments, EvaluationArguments), description="LLMBox description")
    model_args, dataset_args, evaluation_args = parser.parse_args_into_dataclasses()
    check_args(model_args, dataset_args, evaluation_args)
    set_logging(model_args, dataset_args, evaluation_args)

    return model_args, dataset_args, evaluation_args
