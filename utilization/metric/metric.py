from logging import getLogger
from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    from ..dataset import Dataset
    from ..utils import DatasetArguments, EvaluationArguments, ModelArguments

logger = getLogger(__name__)


class Metric:
    r"""The base class for metric calculation."""

    _last_score_lists = {}

    def __init__(self, *args, **kwargs):
        r"""Initialize the metric."""
        pass

    def __call__(self, predictions, references) -> Dict[str, float]:
        r""" Compute specific metric scores between predictions and references.

        Args:
            predictions (List[Union[str, int, float]]): The predicted answers.
            references (List[Union[str, int, float, List[str]]]): The real answers.

        Returns:
            Dict[str, float]: The metric score.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} metric must implement the `__call__` function for score calculation."
        )

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def setup_metric(
        self, model_args: "ModelArguments", dataset_args: "DatasetArguments", evaluation_args: "EvaluationArguments",
        dataset: "Dataset"
    ):
        pass

    @property
    def last_score_lists(self) -> Dict[str, List[float]]:
        if self._last_score_lists is None:
            logger.warning(f"Metric {self.__class__.__name__} have not been called yet. Return empty score lists.")
            return dict()
        return {m: list(l) for m, l in self._last_score_lists.items()}

    @last_score_lists.setter
    def last_score_lists(self, value: Dict[str, List[float]]):
        assert all(
            isinstance(k, (list, np.ndarray)) for k in value.values()
        ), f"Score lists should be a list or np.ndarray. Got {value}"
        self._last_score_lists = value
