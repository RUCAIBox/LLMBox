import os
import re
import sys
from builtins import bool
from copy import copy
from dataclasses import MISSING, dataclass
from logging import getLogger
from typing import ClassVar, Dict, List, Literal, Optional, Set, Tuple, Union

import openai
from transformers.hf_argparser import HfArg, HfArgumentParser

from ..model.enum import ANTHROPIC_MODELS, OPENAI_CHAT_MODELS, OPENAI_MODELS
from .logging import log_levels, set_logging

logger = getLogger(__name__)


def get_redacted(sensitive: Optional[str]) -> str:
    if sensitive is None:
        return ""
    middle = len(sensitive) - 12
    if middle <= 0:
        return "*" * len(sensitive)
    return sensitive[:8] + "*" * middle + sensitive[-4:]


def filter_none_repr(self):
    kwargs = {}
    redact = getattr(self, "_redact", set())
    for key, value in self.__dict__.items():
        if value is not None and not key.startswith("_"):
            kwargs[key] = value if key not in redact else get_redacted(value)
    return f"{self.__class__.__name__}({', '.join(f'{key}={value!r}' for key, value in kwargs.items())})"


@dataclass
class ModelArguments:
    model_name_or_path: str = HfArg(
        default=MISSING,
        aliases=["--model", "-m"],
        help="The model name or path, e.g., davinci-002, meta-llama/Llama-2-7b-hf, ./mymodel",
    )
    model_type: str = HfArg(
        default=None,
        help="The type of the model, which can be chosen from `base` or `instruction`.",
        metadata={"choices": ["base", "instruction"]},
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
    anthropic_api_key: str = HfArg(
        default=None,
        help="The Anthropic API key",
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
        help="Values>1 penalize new tokens based on their existing frequency in the prompt and generated text, vice"
        " versa.",
    )
    presence_penalty: float = HfArg(
        default=None,
        help="Positive values penalize new tokens based on whether they appear in the generated text, vice versa.",
    )
    stop: Union[str, List[str]] = HfArg(
        default=None,
        help="List of strings that stop the generation when they are generated. E.g. --stop 'stop' 'sequence'",
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

    seed: ClassVar[int] = None  # use class variable to facilitate type hint inference

    __repr__ = filter_none_repr

    # redact sensitive information when logging with `__repr__`
    _redact = {"openai_api_key", "anthropic_api_key"}

    # simplify logging with model-specific arguments
    _model_specific_arguments: ClassVar[Dict[str, Set[str]]] = {
        "openai": {"openai_api_key"},
        "anthropic": {"anthropic_api_key"},
        "huggingface": {"device_map", "vllm", "flash_attention", "tokenizer_name_or_path"},
    }

    def is_openai_model(self) -> bool:
        return self._model_impl == "openai"

    def is_anthropic_model(self) -> bool:
        return self._model_impl == "anthropic"

    def is_huggingface_model(self) -> bool:
        return self._model_impl == "huggingface"

    def __post_init__(self):
        # set _model_impl first
        if self.model_name_or_path.lower() in OPENAI_MODELS:
            self._model_impl = "openai"
        elif self.model_name_or_path.lower() in ANTHROPIC_MODELS:
            self._model_impl = "anthropic"
        else:
            self._model_impl = "huggingface"

        # set `self.openai_api_key` and `openai.api_key` from environment variables
        if "OPENAI_API_KEY" in os.environ and self.openai_api_key is None:
            self.openai_api_key = os.environ["OPENAI_API_KEY"]
        if self.openai_api_key is not None:
            openai.api_key = self.openai_api_key
        if self.is_openai_model() and self.openai_api_key is None:
            raise ValueError(
                "OpenAI API key is required. Please set it by passing a `--openai_api_key` or through environment variable `OPENAI_API_KEY`."
            )

        # set `self.anthropic_api_key` from environment variables
        if "ANTHROPIC_API_KEY" in os.environ and self.anthropic_api_key is None:
            self.anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
        if self.is_anthropic_model() and self.anthropic_api_key is None:
            raise ValueError(
                "Anthropic API key is required. Please set it by passing a `--anthropic_api_key` or through environment variable `ANTHROPIC_API_KEY`."
            )

        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path

        if self.is_openai_model() or self.is_anthropic_model():
            self.vllm = False


@dataclass
class DatasetArguments:
    dataset_name: str = HfArg(
        default=MISSING,
        aliases=["-d", "--dataset"],
        help="The name of a dataset or the name(s) of a/several subset(s) in a dataset. Format: 'dataset'"
        " or 'dataset:subset(s)', e.g., copa, race, race:high, or wmt16:en-ro,en-fr",
    )
    subset_names: ClassVar[Set[str]] = set()
    """The name(s) of a/several subset(s) in a dataset, derived from `dataset_name` argument on initalization"""
    dataset_path: Optional[str] = HfArg(
        default=None,
        help="The path of dataset if loading from local. Supports repository cloned from huggingface or "
        "dataset saved by `save_to_disk`.",
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
        aliases=["-fmt"],
        default="{source}{target}",
        help="The format to format the `source` and `target` for each instance",
    )

    num_shots: int = HfArg(
        aliases=["-shots"],
        default=0,
        help="The few-shot number for demonstration",
    )
    ranking_with_options: bool = HfArg(
        default=False,
        help="Whether to evaluate with all options for ranking task",
    )
    ranking_type: Literal["ppl_of_whole_option"] = HfArg(
        default="ppl_of_whole_option",
        help="The evaluation method for ranking task",
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
        default="base",
        help="The method to prompt, eg. 'base', 'least_to_most', 'pal'. Only available for some specific datasets.",
        metadata={"choices": ["base", "least_to_most", "pal"]},
    )
    perspective_api_key: str = HfArg(
        default=None,
        help="The Perspective API key",
    )
    proxy_port: int = HfArg(
        default=None,
        help="The port of the proxy",
    )

    # set in `set_logging` with format "{evaluation_results_dir}/{log_filename}.json"
    evaluation_results_path: ClassVar[str] = None

    __repr__ = filter_none_repr

    def __post_init__(self):
        if ":" in self.dataset_name:
            self.dataset_name, subset_names = self.dataset_name.split(":")
            self.subset_names = set(subset_names.split(","))

        # argparse encodes string with unicode_escape, decode it to normal string, e.g., "\\n" -> "\n"
        self.instance_format = self.instance_format.encode('utf-8').decode('unicode_escape')
        if not self.ranking_with_options and self.ranking_type != "ppl_of_whole_option":
            raise ValueError(
                "The `ranking_type` argument is only available for ranking task with options, "
                "which requires `ranking_with_options` to be True."
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
    log_level: str = HfArg(
        default="info",
        help="Logger level to use on the main node. Possible choices are the log levels as strings: 'debug', 'info', "
        "'warning', 'error' and 'critical'",
        metadata={"choices": log_levels.keys()},
    )
    evaluation_results_dir: str = HfArg(
        default="evaluation_results",
        help="The directory to save evaluation results, which includes source"
        " and target texts, generated texts, and the references.",
    )
    dry_run: bool = HfArg(
        default=False,
        help="Test the evaluation pipeline without actually calling the model.",
    )

    __repr__ = filter_none_repr

    def __post_init__(self):
        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs(self.evaluation_results_dir, exist_ok=True)


def check_args(model_args: ModelArguments, dataset_args: DatasetArguments, evaluation_args: EvaluationArguments):
    r"""Check the validity of arguments.

    Args:
        model_args (ModelArguments): The global configurations.
        dataset_args (DatasetArguments): The dataset configurations.
        evaluation_args (EvaluationArguments): The evaluation configurations.
    """
    model_args.seed = evaluation_args.seed
    if model_args.model_name_or_path.lower() in OPENAI_CHAT_MODELS and dataset_args.batch_size > 1:
        dataset_args.batch_size = 1
        logger.warning(
            f"OpenAI chat-based model {model_args.model_name_or_path} doesn't support batch_size > 1, automatically set batch_size to 1."
        )
    if model_args.model_name_or_path.lower() in ANTHROPIC_MODELS and dataset_args.batch_size > 1:
        dataset_args.batch_size = 1
        logger.warning(
            f"Claude model {model_args.model_name_or_path} doesn't support batch_size > 1, automatically set batch_size to 1."
        )

    if dataset_args.dataset_name == "vicuna_bench" and model_args.openai_api_key is None:
        raise ValueError(
            "OpenAI API key is required for GPTEval metrics. Please set it by passing a `--openai_api_key` or through environment variable `OPENAI_API_KEY`."
        )

    args_ignored = set()
    for model_impl, args in model_args._model_specific_arguments.items():
        if model_impl != model_args._model_impl:
            args_ignored.update(args)
    # some arguments might be shared by multiple model implementations
    args_ignored -= model_args._model_specific_arguments[model_args._model_impl]

    for arg in args_ignored:
        if hasattr(model_args, arg):
            # Ellipsis is just a placeholder that never equals to any default value of the argument
            if model_args.__dataclass_fields__[arg].hash:
                logger.warning(f"Argument `{arg}` is not supported for model `{model_args.model_name_or_path}`")
            setattr(model_args, arg, None)


def parse_argument(args=None) -> Tuple[ModelArguments, DatasetArguments, EvaluationArguments]:
    r"""Parse arguments from command line. Using `argparse` for predefined ones, and an easy mannal parser for others (saved in `kwargs`).

    Returns:
        Namespace: the parsed arguments
    """
    if args is None:
        args = copy(sys.argv[1:])
    parser = HfArgumentParser((ModelArguments, DatasetArguments, EvaluationArguments), description="LLMBox description")
    model_args, dataset_args, evaluation_args = parser.parse_args_into_dataclasses(args)
    commandline_args = {arg.lstrip('-') for arg in args if arg.startswith("-")}
    for type_args in [model_args, dataset_args, evaluation_args]:
        for name, field in type_args.__dataclass_fields__.items():
            field.hash = name in commandline_args  # borrow `hash` attribute to indicate whether the argument is set
    set_logging(model_args, dataset_args, evaluation_args)
    check_args(model_args, dataset_args, evaluation_args)

    # log arguments and environment variables
    redact_dict = {f"--{arg}": get_redacted(getattr(model_args, arg, "")) for arg in model_args._redact}
    for key, value in redact_dict.items():
        if key in args:
            args[args.index(key) + 1] = repr(value)
    logger.info("Command line arguments: {}".format(" ".join(args)))
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.info(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(evaluation_args)

    return model_args, dataset_args, evaluation_args
