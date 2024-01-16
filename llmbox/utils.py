import datetime
import logging
import os
import warnings
from builtins import bool
from dataclasses import MISSING, dataclass
from logging import getLogger
from typing import ClassVar, List, Optional, Set, Tuple, Union

import coloredlogs
import tqdm
from transformers.hf_argparser import HfArg, HfArgumentParser

from .model.enum import OPENAI_CHAT_MODELS

__all__ = ['ModelArguments', 'DatasetArguments', 'EvaluationArguments', 'parse_argument', 'dynamic_interval_tqdm']

logger = getLogger(__name__)

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

DEFAULT_LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'

DEFAULT_DATETIME_FORMAT = '%Y_%m_%d-%H_%M_%S'  # Compatible with windows, which does not support ':' in filename


class dynamic_interval_tqdm(tqdm.tqdm):

    def __init__(self, iterable=None, intervals=None, desc=None, disable=False, unit='it',
                 dynamic_ncols=False, total=None, **kwargs):
        super().__init__(iterable=iterable, desc=desc, disable=disable, unit=unit, unit_scale=False,
                         dynamic_ncols=dynamic_ncols, total=total, **kwargs)
        self.intervals = intervals
        if len(intervals) != total:
            raise ValueError(f"Length of intervals {len(intervals)} does not match total {total}.")

    def __iter__(self):
        """Overwrite the original tqdm iterator to support dynamic intervals."""

        iterable = self.iterable

        if self.disable:
            for obj in iterable:
                yield obj
            return

        mininterval = self.mininterval
        last_print_t = self.last_print_t
        last_print_n = self.last_print_n
        min_start_t = self.start_t + self.delay
        n = self.n
        time = self._time

        try:
            for obj in iterable:
                yield obj
                # Update the progress bar with dynamic intervals
                n += 1 / self.intervals[int(n)]

                if n - last_print_n >= self.miniters:
                    cur_t = time()
                    dt = cur_t - last_print_t
                    if dt >= mininterval and cur_t >= min_start_t:
                        self.update(n - last_print_n)
                        last_print_n = self.last_print_n
                        last_print_t = self.last_print_t
        finally:
            self.n = n
            self.close()


@dataclass
class ModelArguments:

    model_name_or_path: str = HfArg(
        default=MISSING,
        aliases=["--model", "-m"],
        help="The model name or path, e.g., davinci-002, meta-llama/Llama-2-7b-hf, ./mymodel"
    )
    model_type: str = HfArg(
        default=None,
        help="The type of the model, which can be chosen from `base` or `instruction`.",
        metadata={"choices": ['base', 'instruction', None]}
    )
    device_map: str = HfArg(
        default="auto",
        help="The device map for model and data",
    )
    vllm: bool = HfArg(
        default=True,
        help="Whether to use vllm",
    )
    flash_attention: bool = HfArg(
        default=True,
        help="Whether to use flash attention",
    )
    openai_api_key: str = HfArg(
        default=None,
        help="The OpenAI API key",
    )

    tokenizer_name_or_path: str = HfArg(
        default=None, aliases=["--tokenizer"], help="The tokenizer name or path, e.g., meta-llama/Llama-2-7b-hf"
    )

    max_tokens: Optional[int] = HfArg(
        default=None,
        help="The maximum number of tokens for output generation",
    )
    max_length: Optional[int] = HfArg(
        default=None,
        help="The maximum number of tokens of model input sequence",
    )
    temperature: float = HfArg(
        default=None,
        help="The temperature for models",
    )
    top_p: float = HfArg(
        default=None,
        help="The model considers the results of the tokens with top_p probability mass.",
    )
    top_k: float = HfArg(
        default=None,
        help="The model considers the token with top_k probability.",
    )
    frequency_penalty: float = HfArg(
        default=None,
        help="Positive values penalize new tokens based on their existing frequency in the generated text, vice versa.",
    )
    repetition_penalty: float = HfArg(
        default=None,
        help="Values>1 penalize new tokens based on their existing frequency in the prompt and generated text, vice versa.",
    )
    presence_penalty: float = HfArg(
        default=None,
        help="Positive values penalize new tokens based on whether they appear in the generated text, vice versa.",
    )
    stop: Union[str, List[str]] = HfArg(
        default=None,
        help="List of strings that stop the generation when they are generated.",
    )
    no_repeat_ngram_size: int = HfArg(
        default=None,
        help="All ngrams of that size can only occur once.",
    )

    best_of: int = HfArg(
        default=None,
        aliases=["--num_beams"],
        help="The beam size for beam search",
    )
    length_penalty: float = HfArg(
        default=None,
        help="Positive values encourage longer sequences, vice versa. Used in beam search.",
    )
    early_stopping: Union[bool, str] = HfArg(
        default=None,
        help="Positive values encourage longer sequences, vice versa. Used in beam search.",
    )

    def __post_init__(self):
        if "OPENAI_API_KEY" in os.environ and self.openai_api_key is None:
            self.openai_api_key = os.environ["OPENAI_API_KEY"]

        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path


@dataclass
class DatasetArguments:

    dataset_name: str = HfArg(
        default=MISSING,
        aliases=["-d", "--dataset"],
        help="The name of a dataset or the name(s) of a/several subset(s) in a dataset. Format: 'dataset' or 'dataset:subset(s)', e.g., copa, race, race:high, or wmt16:en-ro,en-fr"
    )
    subset_names: ClassVar[Set[str]] = set()
    """The name(s) of a/several subset(s) in a dataset, derived from `dataset_name` argument on initalization"""
    dataset_path: Optional[str] = HfArg(
        default=None,
        help="The path of dataset if loading from local. Supports repository cloned from huggingface or dataset saved by `save_to_disk`."
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
        aliases=["-bsz", "-b"],
        help="The evaluation batch size",
    )
    sample_num: int = HfArg(
        default=1,
        aliases=["--majority", "--consistency"],
        help="The sampling number for self-consistency",
    )

    kate: bool = HfArg(default=False, aliases=["-kate"], help="Whether to use KATE as an ICL strategy")
    globale: bool = HfArg(default=False, aliases=["-globale"], help="Whether to use GlobalE as an ICL strategy")
    ape: bool = HfArg(default=False, aliases=["-ape"], help="Whether to use APE as an ICL strategy")
    cot: str = HfArg(
        default='base',
        help="The method to prompt, eg. 'base', 'least_to_most', 'pal'. Only available for some specific datasets.",
        metadata={"choices": ['base', 'least_to_most', 'pal']},
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
    log_level: str = HfArg(
        default="info",
        help="Logger level to use on the main node. Possible choices are the log levels as strings: 'debug', 'info', 'warning', 'error' and 'critical'",
        metadata={"choices": log_levels.keys()},
    )
    evaluation_results_dir: str = HfArg(
        default="evaluation_results",
        help="The directory to save evaluation results, which includes source"
        " and target texts, generated texts, and the references.",
    )

    def __post_init__(self):
        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs(self.evaluation_results_dir, exist_ok=True)


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
    log_filename = f"{model_name}-{dataset_name}-{num_shots}shot-{execution_time}"
    log_path = f"{evaluation_args.logging_dir}/{log_filename}.log"
    evaluation_results_path = f"{evaluation_args.evaluation_results_dir}/{log_filename}.json"
    dataset_args.evaluation_results_path = evaluation_results_path  # type: ignore

    # add file handler to root logger
    handler = logging.FileHandler(log_path)
    formatter = coloredlogs.BasicFormatter(fmt=DEFAULT_LOG_FORMAT)
    coloredlogs.HostNameFilter.install(handler=handler)
    handler.setLevel(level=log_levels[file_log_level])
    handler.setFormatter(formatter)
    package_logger.addHandler(handler)

    # finish logging initialization
    logger.info(f"Saving logs to {log_path}")


def check_args(model_args, dataset_args, evaluation_args):
    r"""Check the validity of arguments.

    Args:
        model_args (ModelArguments): The global configurations.
        dataset_args (DatasetArguments): The dataset configurations.
        evaluation_args (EvaluationArguments): The evaluation configurations.
    """
    model_args.seed = evaluation_args.seed
    if model_args.model_name_or_path.lower() in OPENAI_CHAT_MODELS and dataset_args.batch_size > 1:
        dataset_args.batch_size = 1
        warnings.warn(
            f"OpenAI chat-based model {model_args.model_name_or_path} doesn't support batch_size > 1, automatically set batch_size=1."
        )


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
