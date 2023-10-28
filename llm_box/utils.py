from argparse import ArgumentParser
from dataclasses import MISSING, dataclass
from typing import Tuple

from transformers.hf_argparser import HfArg, HfArgumentParser


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


def parse_argument() -> Tuple[ModelArguments, DatasetArguments, EvaluationArguments]:
    r"""Parse arguments from command line. Using `argparse` for predefined ones, and an easy mannal parser for others (saved in `kwargs`).

    Returns:
        Namespace: the parsed arguments
    """
    ArgumentParser()
    parser = HfArgumentParser((ModelArguments, DatasetArguments, EvaluationArguments), description="LLMBox description")
    model_args, dataset_args, evaluation_args = parser.parse_args_into_dataclasses()

    return model_args, dataset_args, evaluation_args
