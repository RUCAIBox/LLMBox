import argparse
import json
import os
import re
import sys
import typing
from copy import copy
from dataclasses import MISSING, dataclass
from logging import getLogger
from typing import Callable, ClassVar, Dict, List, Literal, Optional, Set, Tuple, Union

import tiktoken
from transformers import BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.hf_argparser import HfArg, HfArgumentParser

from ..dataset.dataset_enum import DEFAULT_VLLM_DATASETS
from ..model.model_enum import (ANTHROPIC_CHAT_COMPLETIONS_ARGS, API_MODELS, DASHSCOPE_CHAT_COMPLETIONS_ARGS,
                                QIANFAN_CHAT_COMPLETIONS_ARGS)
from .logging import filter_none_repr, get_redacted, list_datasets, log_levels, passed_in_commandline, set_logging

logger = getLogger(__name__)

ENVIRONMENT_ARGUMENTS = {"CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "TOKENIZERS_PARALLELISM"}

if typing.TYPE_CHECKING:
    batch_size_type = int
else:
    batch_size_type = str

LOADER = Callable[["ModelArguments"], Tuple["PreTrainedModel", Union["PreTrainedTokenizer",
                                                                         "PreTrainedTokenizerFast"]]]


class ModelBackendMixin:

    model_backend: Literal["anthropic", "dashscope", "huggingface", "openai", "qianfan", "vllm"]

    def is_openai_model(self) -> bool:
        return self.model_backend == "openai"

    def is_anthropic_model(self) -> bool:
        return self.model_backend == "anthropic"

    def is_dashscope_model(self) -> bool:
        return self.model_backend == "dashscope"

    def is_qianfan_model(self) -> bool:
        return self.model_backend == "qianfan"

    def is_huggingface_model(self) -> bool:
        return self.model_backend == "huggingface"

    def is_vllm_model(self) -> bool:
        return self.model_backend == "vllm"

    def is_local_model(self) -> bool:
        """Backed by Huggingface or vLLM model."""
        return self.is_huggingface_model() or self.is_vllm_model()


@dataclass
class ModelArguments(ModelBackendMixin):
    model_name_or_path: str = HfArg(
        default=MISSING,
        aliases=["--model", "-m"],
        help="The model name or path, e.g., davinci-002, meta-llama/Llama-2-7b-hf, ./mymodel",
    )
    model_type: Literal["base", "instruction", "chat"] = HfArg(
        default=None,
        help="The type of the model",
    )
    model_backend: Literal["anthropic", "dashscope", "huggingface", "openai", "qianfan", "vllm"] = HfArg(
        default=None,
        help="The model backend",
    )
    device_map: str = HfArg(
        default="auto",
        help="The device map for model and data",
    )
    prefix_caching: bool = HfArg(
        default=None,
        help="Whether to cache prefix in get_ppl mode",
    )
    vllm: bool = HfArg(
        default=None,
        help="Whether to use vllm",
    )
    flash_attention: bool = HfArg(
        aliases=["--flash_attn"],
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
    dashscope_api_key: str = HfArg(
        default=None,
        help="The Dashscope API key",
    )
    qianfan_access_key: str = HfArg(
        default=None,
        help="The Qianfan access key",
    )
    qianfan_secret_key: str = HfArg(
        default=None,
        help="The Qianfan secret key",
    )
    api_endpoint: Optional[str] = HfArg(
        default=None,
        help="The API endpoint",
    )

    tokenizer_name_or_path: str = HfArg(
        default=None,
        aliases=["--tokenizer"],
        help="The tokenizer name or path, e.g., cl100k_base, meta-llama/Llama-2-7b-hf, ./mymodel"
    )

    max_tokens: Optional[int] = HfArg(
        default=None,
        aliases=["--max_new_tokens"],  # compatible with HF arguments
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
        aliases=["--num_beams"],  # compatible with HF arguments
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

    system_prompt: Optional[str] = HfArg(
        aliases=["-sys"],
        default=None,
        help="The system prompt for chat-based models",
    )
    chat_template: Optional[str] = HfArg(
        default=None,
        help="The chat template for local chat-based models",
    )

    bnb_config: Optional[str] = HfArg(default=None, help="JSON string for BitsAndBytesConfig parameters.")

    load_in_8bit: bool = HfArg(
        default=False,
        help="Whether to use bnb's 8-bit quantization to load the model.",
    )

    load_in_4bit: bool = HfArg(
        default=False,
        help="Whether to use bnb's 4-bit quantization to load the model.",
    )

    gptq: bool = HfArg(
        default=False,
        help="Whether the model is a gptq quantized model.",
    )

    vllm_gpu_memory_utilization: float = HfArg(
        aliases=["--vllm_mem"],
        default=None,
        help="The maximum gpu memory utilization of vllm.",
    )

    torch_dtype: Literal["float16", "bfloat16", "float32", "auto"] = HfArg(
        default="float16",
        help="The torch dtype for model input and output",
    )

    seed: ClassVar[int] = None  # use class variable to facilitate type hint inference

    load_hf_model: ClassVar[Optional["LOADER"]] = None

    _argument_group_name = "model arguments"

    __repr__ = filter_none_repr

    passed_in_commandline = passed_in_commandline

    # redact sensitive information when logging with `__repr__`
    _redact = {"openai_api_key", "anthropic_api_key", "dashscope_api_key", "qianfan_access_key", "qianfan_secret_key"}

    # simplify logging with model-specific arguments
    _model_specific_arguments: ClassVar[Dict[str, Set[str]]] = {
        "anthropic": {"anthropic_api_key"} | set(ANTHROPIC_CHAT_COMPLETIONS_ARGS),
        "dashscope": {"dashscope_api_key"} | set(DASHSCOPE_CHAT_COMPLETIONS_ARGS),
        "openai": set(),  # openai model is used for gpt-eval metrics, not specific arguments
        "qianfan": {"qianfan_access_key", "qianfan_secret_key"} | set(QIANFAN_CHAT_COMPLETIONS_ARGS),
        "vllm": {"vllm", "prefix_caching", "flash_attention", "gptq", "vllm_gpu_memory_utilization", "chat_template"},
        "huggingface": {
            "device_map", "vllm", "prefix_caching", "flash_attention", "bnb_config", "load_in_8bit", "load_in_4bit",
            "gptq", "chat_template", "stop"
        },
    }

    def __post_init__(self):
        if self.vllm is None:
            self.vllm = not self.prefix_caching

        # set _model_impl first
        if self.model_backend is None:
            if self.model_name_or_path in API_MODELS:
                self.model_backend = API_MODELS[self.model_name_or_path]["model_backend"]
            elif self.vllm:
                self.model_backend = "vllm"
            else:
                self.model_backend = "huggingface"

        # set `self.openai_api_key` and `openai.api_key` from environment variables
        if "OPENAI_API_KEY" in os.environ and self.openai_api_key is None:
            self.openai_api_key = os.environ["OPENAI_API_KEY"]
        if self.openai_api_key is not None:
            import openai
            openai.api_key = self.openai_api_key
        if self.is_openai_model():
            if self.openai_api_key is None:
                raise ValueError(
                    "OpenAI API key is required. Please set it by passing a `--openai_api_key` or through environment variable `OPENAI_API_KEY`."
                )
            if self.tokenizer_name_or_path is None:
                self.tokenizer_name_or_path = tiktoken.encoding_name_for_model(self.model_name_or_path)

        # set `self.anthropic_api_key` from environment variables
        if "ANTHROPIC_API_KEY" in os.environ and self.anthropic_api_key is None:
            self.anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
        if self.is_anthropic_model():
            if self.anthropic_api_key is None:
                raise ValueError(
                    "Anthropic API key is required. Please set it by passing a `--anthropic_api_key` or through environment variable `ANTHROPIC_API_KEY`."
                )
            if self.tokenizer_name_or_path is None:
                self.tokenizer_name_or_path = "cl100k_base"

        # set `self.dashscope_api_key` from environment variables
        if "DASHSCOPE_API_KEY" in os.environ and self.dashscope_api_key is None:
            self.dashscope_api_key = os.environ["DASHSCOPE_API_KEY"]
        if self.is_dashscope_model():
            if self.dashscope_api_key is None:
                raise ValueError(
                    "Dashscope API key is required. Please set it by passing a `--dashscope_api_key` or through environment variable `DASHSCOPE_API_KEY`."
                )
            if self.tokenizer_name_or_path is None:
                self.tokenizer_name_or_path = self.model_name_or_path

        # set `self.qianfan_access_key` and `self.qianfan_secret_key` from environment variables
        if "QIANFAN_ACCESS_KEY" in os.environ and self.qianfan_access_key is None:
            self.qianfan_access_key = os.environ["QIANFAN_ACCESS_KEY"]
        if "QIANFAN_SECRET_KEY" in os.environ and self.qianfan_secret_key is None:
            self.qianfan_secret_key = os.environ["QIANFAN_SECRET_KEY"]
        if self.is_qianfan_model():
            if self.qianfan_access_key is None or self.qianfan_secret_key is None:
                raise ValueError(
                    "Qianfan API access key and secret key is required. Please set it by passing `--qianfan_access_key` and `--qianfan_secret_key` or through environment variable `QIANFAN_ACCESS_KEY` and `QIANFAN_SECRET_KEY`."
                )
            if self.tokenizer_name_or_path is None:
                self.tokenizer_name_or_path = "cl100k_base"

        if self.is_local_model():
            if self.model_type == "chat":
                if not re.search(r"chat|instruct", self.model_name_or_path.lower()):
                    logger.warning(
                        f"Model {self.model_name_or_path} seems to be a base model, you can set --model_type to `base` or `instruction` to use base format."
                    )
            else:
                if re.search(r"chat|instruct", self.model_name_or_path.lower()):
                    logger.warning(
                        f"Model {self.model_name_or_path} seems to be a chat-based model, you can set --model_type to `chat` to use chat format."
                    )

        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path

        if self.model_name_or_path in API_MODELS:
            auto_model_type = API_MODELS[self.model_name_or_path]["model_type"]
        else:
            auto_model_type = None

        if self.model_type is None and auto_model_type is not None:
            self.model_type = auto_model_type
        elif self.model_type is None and auto_model_type is None:
            self.model_type = "base" if self.is_local_model() else "instruction"
        elif auto_model_type is not None and self.model_type != auto_model_type:
            logger.warning(
                f"Model {self.model_name_or_path} seems to be a {auto_model_type} model, but get model_type {self.model_type}."
            )

        if self.model_name_or_path in API_MODELS:
            auto_endpoint = API_MODELS[self.model_name_or_path]["endpoint"]
        elif not self.is_local_model():
            auto_endpoint = "chat/completions"
        else:
            auto_endpoint = None

        if self.api_endpoint is None:
            self.api_endpoint = auto_endpoint

        # try to load as vllm model. If failed, fallback to huggingface model.
        # See `model/load.py` for details.
        if not self.is_local_model():
            self.vllm = False
        elif self.is_vllm_model():
            self.vllm = True

        if self.vllm:
            self.vllm_gpu_memory_utilization = 0.9

        if self.prefix_caching is None:
            # prefix caching of vllm is still experimental
            self.prefix_caching = not self.vllm

        if self.model_type != "chat":
            if self.system_prompt:
                raise ValueError(
                    "The system_prompt is only available for chat-based model. Please use a chat model and set `--model_type chat`."
                )
            if self.chat_template:
                raise ValueError(
                    "The chat_template is only available for huggingface chat-based model. Please use a chat model and set `--model_type chat`."
                )

        # argparse encodes string with unicode_escape, decode it to normal string, e.g., "\\n" -> "\n"
        if self.stop is not None:
            if isinstance(self.stop, str):
                self.stop = [self.stop]
            for idx in range(len(self.stop)):
                self.stop[idx] = self.stop[idx].encode('utf-8').decode('unicode_escape')


@dataclass
class DatasetArguments:
    dataset_names: List[str] = HfArg(
        default=MISSING,
        aliases=["-d", "--dataset"],
        help=
        "Space splitted dataset names. If only one dataset is specified, it can be followed by subset names or category names. Format: 'dataset1 dataset2', 'dataset:subset1,subset2', or 'dataset:[cat1],[cat2]', e.g., 'copa race', 'race:high', 'wmt16:en-ro,en-fr', or 'mmlu:[stem],[humanities]'. Supported datasets: "
        + ", ".join(list_datasets()),
        metadata={"metavar": "DATASET"},
    )
    subset_names: ClassVar[Set[str]] = set()
    """The name(s) of a/several subset(s) in a dataset, derived from `dataset_names` argument on initalization"""
    batch_size: batch_size_type = HfArg(
        default=1,
        aliases=["-bsz", "-b"],
        help=
        "The evaluation batch size. Specify an integer (e.g., '10') to use a fixed batch size for all iterations. Alternatively, append ':auto' (e.g., '10:auto') to start with the specified batch size and automatically adjust it in subsequent iterations to maintain constant CUDA memory usage",
    )
    auto_batch_size: ClassVar[bool] = False
    dataset_path: Optional[str] = HfArg(
        default=None,
        help="The path of dataset if loading from local. Supports repository cloned from huggingface, "
        "dataset saved by `save_to_disk`, or a template string e.g. 'mmlu/{split}/{subset}_{split}.csv'.",
    )

    evaluation_set: Optional[str] = HfArg(
        default=None,
        help="The set name for evaluation, supporting slice, e.g., validation, test, validation[:10]",
    )
    example_set: Optional[str] = HfArg(
        default=None,
        help="The set name for demonstration, supporting slice, e.g., train, dev, train[:10]",
    )

    instruction: Optional[str] = HfArg(
        default=None,
        help="The format to format the `source` and `target` for each instance",
    )

    num_shots: int = HfArg(
        aliases=["-shots", "--max_num_shots"],
        default=0,
        help="The maximum few-shot number for demonstration",
    )
    max_example_tokens: int = HfArg(
        default=1024,
        help="The maximum token number of demonstration",
    )

    ranking_type: Literal["generation", "ppl", "prob", "ppl_no_option"] = HfArg(
        default="ppl_no_option",
        help="The evaluation and prompting method for ranking task",
    )
    sample_num: int = HfArg(
        default=1,
        aliases=["--majority", "--consistency"],
        help="The sampling number for self-consistency",
    )
    kate: bool = HfArg(default=False, aliases=["-kate"], help="Whether to use KATE as an ICL strategy")
    globale: bool = HfArg(default=False, aliases=["-globale"], help="Whether to use GlobalE as an ICL strategy")
    ape: bool = HfArg(default=False, aliases=["-ape"], help="Whether to use APE as an ICL strategy")
    cot: Optional[Literal["base", "least_to_most", "pal", "retrieval", "retrieval_content"]] = HfArg(
        default=None,
        help="The method to prompt. Only available for some specific datasets (e.g., GSM8K, GPQA).",
    )
    perspective_api_key: str = HfArg(
        default=None,
        help="The Perspective API key for toxicity metrics",
    )
    pass_at_k: int = HfArg(
        default=None,
        help="The k value for pass@k metric",
    )
    max_evaluation_instances: int = HfArg(
        aliases=["-i"],
        default=0,
        help="The maximum number of evaluation instances per dataset (subset)",
    )
    shuffle_choices: bool = HfArg(
        default=False,
        help="Whether to shuffle the choices for ranking task",
    )

    continue_from: ClassVar[int] = 0
    proxy_port: ClassVar[int] = None
    dataset_threading: ClassVar[bool] = True

    # set in `set_logging` with format "{evaluation_results_dir}/{log_filename}.json"
    evaluation_results_path: ClassVar[Optional[str]] = None

    _argument_group_name = "dataset arguments"

    __repr__ = filter_none_repr

    passed_in_commandline = passed_in_commandline

    def __post_init__(self):
        for d in self.dataset_names:
            if ":" in d:
                if len(self.dataset_names) > 1:
                    raise ValueError("Only one dataset can be specified when using subset names.")
                self.dataset_names[0], subset_names = d.split(":")
                self.subset_names = set(subset_names.split(","))

        # argparse encodes string with unicode_escape, decode it to normal string, e.g., "\\n" -> "\n"
        if isinstance(self.instruction, str):
            self.instruction = self.instruction.encode('utf-8').decode('unicode_escape')

        if isinstance(self.batch_size, str):
            if self.batch_size.endswith(":auto") and self.batch_size[:-len(":auto")].isdigit():
                self.batch_size = int(self.batch_size[:-len(":auto")])
                self.auto_batch_size = True
            elif self.batch_size.isdigit():
                self.batch_size = int(self.batch_size)
            else:
                raise ValueError(
                    f"Invalid batch size: {self.batch_size}. Specify an integer (e.g., '10') to use a fixed batch size for all iterations. Alternatively, append ':auto' (e.g., '10:auto') to start with the specified batch size and automatically adjust it in subsequent iterations to maintain constant CUDA memory usage"
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
        help="The directory to save evaluation results. This includes detailed information for each instance,"
        " such as the input data, reference data, raw results, and processed results.",
    )
    log_results: bool = HfArg(
        default=True,
        help=
        "Whether to log the evaluation results. Notes that the generated JSON file will be the same size as the evaluation dataset itself",
    )
    dry_run: bool = HfArg(
        default=False,
        help="Test the evaluation pipeline without actually calling the model.",
    )
    proxy_port: int = HfArg(
        default=None,
        help="The port of the proxy",
    )
    cuda_visible_devices: Optional[str] = HfArg(
        default=None,
        aliases=["--cuda"],
        help="Override the CUDA_VISIBLE_DEVICES environment variable",
    )
    dataset_threading: bool = HfArg(default=True, help="Load dataset with threading")
    dataloader_workers: int = HfArg(default=0, help="The number of workers for dataloader")
    continue_from: Optional[str] = HfArg(
        default=None,
        help="The path to the evaluation results to continue from",
    )

    _argument_group_name = "evaluation arguments"

    __repr__ = filter_none_repr

    passed_in_commandline = passed_in_commandline

    def __post_init__(self):
        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs(self.evaluation_results_dir, exist_ok=True)
        if self.proxy_port is not None:
            try:
                import httpx
                import openai

                openai.http_client = httpx.Client(proxies=f"http://localhost:{self.proxy_port}")
            except Exception:
                pass


def check_args(model_args: ModelArguments, dataset_args: DatasetArguments, evaluation_args: EvaluationArguments):
    r"""Check the validity of arguments.

    Args:
        model_args (ModelArguments): The global configurations.
        dataset_args (DatasetArguments): The dataset configurations.
        evaluation_args (EvaluationArguments): The evaluation configurations.
    """
    # vllm still has some bugs in ranking task
    if model_args.is_local_model() and all(d not in DEFAULT_VLLM_DATASETS for d in dataset_args.dataset_names) and not model_args.passed_in_commandline("vllm"):
        model_args.vllm = False
        model_args.model_backend = "huggingface"

    # copy arguments
    if evaluation_args.proxy_port:
        dataset_args.proxy_port = evaluation_args.proxy_port

    dataset_args.dataset_threading = evaluation_args.dataset_threading

    model_args.seed = int(evaluation_args.seed)

    if dataset_args.batch_size == 1:
        model_args.prefix_caching = False

    # check models
    if model_args.model_name_or_path in API_MODELS and API_MODELS[
        model_args.model_name_or_path]["model_type"] == "instruction" and dataset_args.batch_size > 1:
        dataset_args.batch_size = 1
        logger.warning(
            f"chat/completions endpoint model {model_args.model_name_or_path} doesn't support batch_size > 1, automatically set batch_size to 1."
        )

    # vllm has its own prefix caching mechanism
    if model_args.prefix_caching and "expandable_segments" not in os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF", ""
    ) and model_args.is_huggingface_model():
        logger.warning(
            f"Prefix caching might results in cuda memory fragmentation, which can be mitigated by setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. See https://pytorch.org/docs/stable/notes/cuda.html#environment-variables for details."
        )

    if evaluation_args.cuda_visible_devices:
        if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != evaluation_args.cuda_visible_devices:
            logger.warning(f"Override CUDA_VISIBLE_DEVICES from {os.environ['CUDA_VISIBLE_DEVICES']} to {evaluation_args.cuda_visible_devices}.")
        os.environ["CUDA_VISIBLE_DEVICES"] = evaluation_args.cuda_visible_devices

    # check dataset
    if "vicuna_bench" in dataset_args.dataset_names and model_args.openai_api_key is None:
        raise ValueError(
            "OpenAI API key is required for GPTEval metrics. Please set it by passing a `--openai_api_key` or through environment variable `OPENAI_API_KEY`."
        )

    if "story_cloze" in dataset_args.dataset_names and dataset_args.dataset_path is None:
        raise ValueError(
            "Story Cloze dataset requires manual download. View details at https://github.com/RUCAIBox/LLMBox/blob/main/utilization/README.md#supported-datasets."
        )

    if "coqa" in dataset_args.dataset_names and dataset_args.dataset_path is None:
        raise ValueError(
            "CoQA dataset requires manual download. View details at https://github.com/RUCAIBox/LLMBox/blob/main/utilization/README.md#supported-datasets."
        )

    if dataset_args.instruction and "{" not in dataset_args.instruction:
        logger.warning("Instruction does not include any variable, so the input remains unchanged across the insatnces. Try to use f-string or jinja2 format to include variables like `{source}` or `{problem}`. See dataset documentation for details.")

    if evaluation_args.dry_run and model_args.prefix_caching:
        model_args.prefix_caching = False

    args_ignored = set()
    for model_impl, args in model_args._model_specific_arguments.items():
        if model_impl != model_args.model_backend:
            args_ignored.update(args)
    # some arguments might be shared by multiple model implementations
    args_ignored -= model_args._model_specific_arguments.get(model_args.model_backend, set())

    for arg in args_ignored:
        if hasattr(model_args, arg):
            # Ellipsis is just a placeholder that never equals to any default value of the argument
            if model_args.__dataclass_fields__[arg].hash:
                logger.warning(f"Argument `{arg}` is not supported for model `{model_args.model_name_or_path}`")
            setattr(model_args, arg, None)


DESCRIPTION_STRING = r"""LLMBox is a comprehensive library for implementing LLMs, including a unified training pipeline and comprehensive model evaluation. LLMBox is designed to be a one-stop solution for training and utilizing LLMs. Through a pratical library design, we achieve a high-level of flexibility and efficiency in both training and utilization stages.
GitHub: https://github.com/RUCAIBox/LLMBox"""

EXAMPLE_STRING = r"""example:
  Evaluating davinci-002 on HellaSwag:
       python inference.py -m davinci-002 -d hellaswag
  Evaluating Gemma on MMLU:
       python inference.py -m gemma-7b -d mmlu -shots 5
  Evaluating Phi-2 on GSM8k using self-consistency and 4-bit quantization:
       python inference.py -m microsoft/phi-2 -d gsm8k -shots 8 --sample_num 100 --load_in_4bit

"""


def parse_argument(args: Optional[List[str]] = None,
                   initalize: bool = True) -> Tuple[ModelArguments, DatasetArguments, EvaluationArguments]:
    r"""Parse arguments from command line. Using `argparse` for predefined ones, and an easy mannal parser for others (saved in `kwargs`).

    Returns:
        Namespace: the parsed arguments
    """
    if args is None:
        args = copy(sys.argv[1:])
    parser = HfArgumentParser(
        (ModelArguments, DatasetArguments, EvaluationArguments),
        description=DESCRIPTION_STRING,
        epilog=EXAMPLE_STRING,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    model_args, dataset_args, evaluation_args = parser.parse_args_into_dataclasses(args)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except (ImportError, ModuleNotFoundError):
        pass

    if model_args.bnb_config:
        bnb_config_dict = json.loads(model_args.bnb_config)
        model_args.bnb_config = BitsAndBytesConfig(**bnb_config_dict)

    commandline_args = {arg.lstrip('-') for arg in args if arg.startswith("-")}
    for type_args in [model_args, dataset_args, evaluation_args]:
        for name, field in type_args.__dataclass_fields__.items():
            field.hash = name in commandline_args  # borrow `hash` attribute to indicate whether the argument is set
    if initalize:
        set_logging(model_args, dataset_args, evaluation_args)
        check_args(model_args, dataset_args, evaluation_args)

    # log arguments and environment variables
    redact_dict = {f"--{arg}": get_redacted(getattr(model_args, arg, "")) for arg in model_args._redact}
    for key, value in redact_dict.items():
        if key in args:
            args[args.index(key) + 1] = repr(value)
    env_args = {arg: os.environ.get(arg, None) for arg in ENVIRONMENT_ARGUMENTS}
    env_args = " ".join(f"{key}={value}" for key, value in env_args.items() if value is not None)
    cmd_args = " ".join(args)
    pid = os.getpid()
    logger.info(f"Run commands (pid={pid}): {env_args} {sys.executable} inference.py {cmd_args}")
    logger.info(evaluation_args)

    return model_args, dataset_args, evaluation_args
