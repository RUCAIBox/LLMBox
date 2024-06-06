import os

# Disable download counts for transformers to accelerate
os.environ["HF_UPDATE_DOWNLOAD_COUNTS"] = "FALSE"

from typing import TYPE_CHECKING, Callable, Optional

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
) -> "Evaluator":
    from .evaluator import Evaluator

    return Evaluator(
        model_args=model_args,
        dataset_args=dataset_args,
        evaluation_args=evaluation_args,
        initalize=initalize,
        load_hf_model=load_hf_model,
    )


__all__ = ["get_evaluator", "parse_argument", "ModelArguments", "DatasetArguments", "EvaluationArguments"]
