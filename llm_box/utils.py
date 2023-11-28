import os
import logging
from builtins import bool
from dataclasses import MISSING, dataclass
from logging import getLogger
from typing import Optional, Tuple, TypeVar, ClassVar, Set
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

DEFAULT_DATETIME_FORMAT = '%Y_%m_%d-%H_%M_%S'  # Compatible with windows, which does not support ':' in filename


@property
def NotImplementedField(self):
    raise NotImplementedError(f"{self.__class__.__name__} has not implemented field.")


@dataclass
class ModelArguments:

    model_name_or_path: str = HfArg(
        default=MISSING, aliases=["--model", "-m"], help="The model name or path, e.g., cuire, llama"
    )
    openai_api_key: str = HfArg(
        default=None,
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
    temperature: float = HfArg(
        default=0,
        help="The temperature for models",
    )
    max_tokens: int = HfArg(
        default=2048,
        help="The maximum number of tokens for output generation",
    )

    def __post_init__(self):
        if "OPENAI_API_KEY" in os.environ and self.openai_api_key is None:
            self.openai_api_key = os.environ["OPENAI_API_KEY"]


@dataclass
class DatasetArguments:

    dataset_name: str = HfArg(
        default=MISSING,
        aliases=["-d", "--dataset"],
        help=
        "The name of a dataset or the name of a subset in a dataset. Format: 'dataset' or 'dataset:subset'. E.g., copa, gsm, race, or race:high"
    )
    """The name of a dataset without subset name. Refer to `subset_names` for the subset name."""

    subset_names: ClassVar[Set[str]] = set()
    """The name of a subset in a dataset, derived from `dataset_name` argument on initalization"""

    dataset_path: Optional[str] = HfArg(
        default=None,
        help=
        "The path of dataset if loading from local. Supports repository cloned from huggingface or dataset saved by `save_to_disk`."
    )
    evaluation_set: Optional[str] = HfArg(
        default=None,
        help="The set name for evaluation, supporting slice, e.g., validation, test, validation[:10]",
    )
    example_set: Optional[str] = HfArg(
        default=None,
        help="The set name for demonstration, supporting slice, e.g., train, dev, train[:10]",
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
    sample_num: int = HfArg(
        default=1,
        help="The path number for sampling for self-consistency",
    )

    kate: bool = HfArg(default=False, aliases=["-kate"], help="Whether to use KATE")
    globale: bool = HfArg(default=False, aliases=["-globale"], help="Whether to use KATE")
    ape: bool = HfArg(default=False, aliases=["-ape"], help="Whether to use KATE")

    use_pal: bool = HfArg(
        default=False,
        help="Whether to use PaL(Program-aided Language Models) to solve problems. Only available for some specific datasets.",
    )



    def __post_init__(self):
        if ":" in self.dataset_name:
            self.dataset_name, subset_names = self.dataset_name.split(":")
            self.subset_names = set(subset_names.split(","))


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
        "Logger level to use on the main node. Possible choices are the log levels as strings: 'debug', 'info', 'warning', 'error' and 'critical'",
        metadata={"choices": log_levels.keys()},
    )

    def __post_init__(self):
        if not os.path.exists(self.logging_dir):
            os.makedirs(self.logging_dir)


def set_logging(
    model_args: ModelArguments,
    dataset_args: DatasetArguments,
    evaluation_args: EvaluationArguments,
    file_log_level: str = 'info',
) -> None:
    """Set the logging level for standard output and file."""

    # Use package logger to disable logging of other packages. Set the level to DEBUG first
    # to allow all logs from our package, and then set the level to the desired one.
    package_logger = logging.getLogger(__package__)
    coloredlogs.install(
        level=logging.DEBUG,
        logger=package_logger,
        fmt=DEFAULT_LOG_FORMAT,
    )
    package_logger.handlers[0].setLevel(level=log_levels[evaluation_args.log_level])

    # set the log file
    model_name = model_args.model_name_or_path.strip("/").split("/")[-1]
    dataset_name = dataset_args.dataset_name + (
        "_" + ",".join(dataset_args.subset_names) if dataset_args.subset_names else ""
    )
    num_shots = str(dataset_args.num_shots)
    execution_time = datetime.datetime.now().strftime(DEFAULT_DATETIME_FORMAT)
    log_filename = f"{model_name}-{dataset_name}-{num_shots}-{execution_time}.log"
    log_path = f"{evaluation_args.logging_dir}/{log_filename}"

    # add file handler to root logger
    handler = logging.FileHandler(log_path)
    formatter = coloredlogs.BasicFormatter(fmt=DEFAULT_LOG_FORMAT)
    coloredlogs.HostNameFilter.install(handler=handler)
    handler.setLevel(level=log_levels[file_log_level])
    handler.setFormatter(formatter)
    package_logger.addHandler(handler)

    # finish logging initialization
    logger.warning(f"Saving logs to {log_path}")


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


def parse_argument(args=None) -> Tuple[ModelArguments, DatasetArguments, EvaluationArguments]:
    r"""Parse arguments from command line. Using `argparse` for predefined ones, and an easy mannal parser for others (saved in `kwargs`).

    Returns:
        Namespace: the parsed arguments
    """
    parser = HfArgumentParser((ModelArguments, DatasetArguments, EvaluationArguments), description="LLMBox description")
    model_args, dataset_args, evaluation_args = parser.parse_args_into_dataclasses(args)
    check_args(model_args, dataset_args, evaluation_args)
    set_logging(model_args, dataset_args, evaluation_args)

    return model_args, dataset_args, evaluation_args
