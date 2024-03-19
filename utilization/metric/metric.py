from logging import getLogger
from typing import Dict, List

logger = getLogger(__name__)


class Metric:
    r"""The base class for metric calculation."""

    _last_score_lists = {}

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

    def last_score_lists(self) -> Dict[str, List[float]]:
        if self._last_score_lists is None:
            logger.warning(f"Metric {self.__class__.__name__} have not been called yet. Return empty score lists.")
            return dict()
        return {m: list(l) for m, l in self._last_score_lists.items()}
