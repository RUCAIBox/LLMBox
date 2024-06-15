import os

# Disable download counts for transformers to accelerate
os.environ["HF_UPDATE_DOWNLOAD_COUNTS"] = "FALSE"
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

# this file only initializes .utils modules to avoid early import of torch
from .load_model import register_model
from .utils import DatasetArguments, EvaluationArguments, ModelArguments, parse_argument

if TYPE_CHECKING:
    from .evaluator import Evaluator


def get_evaluator(
    *,
    model_args: "ModelArguments",
    dataset_args: "DatasetArguments",
    evaluation_args: Optional["EvaluationArguments"] = None,
    initalize: bool = True,
    load_hf_model: Optional[Callable] = None,
    evaluation_data: Optional[List[Dict[str, Any]]] = None,
    example_data: Optional[List[Dict[str, Any]]] = None,
) -> "Evaluator":
    from .evaluator import Evaluator

    return Evaluator(
        model_args=model_args,
        dataset_args=dataset_args,
        evaluation_args=evaluation_args,
        initalize=initalize,
        load_hf_model=load_hf_model,
        evaluation_data=evaluation_data,
        example_data=example_data,
    )


def register_dataset(name: str):
    """Decorator to register a dataset class to the dataset registry."""

    from .load_dataset import REGISTERY, _validate_dataset_class

    def _register_dataset_class(cls):
        assert _validate_dataset_class(cls), f"{cls} is not a valid dataset class."
        REGISTERY[name] = cls
        return cls

    return _register_dataset_class


__all__ = [
    "get_evaluator", "parse_argument", "ModelArguments", "DatasetArguments", "EvaluationArguments", "register_dataset",
    "register_model"
]
