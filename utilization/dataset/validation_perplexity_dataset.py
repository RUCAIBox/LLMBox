from functools import cached_property
from typing import Any, List, Tuple, Union

import numpy as np

from .dataset import Dataset

__all__ = ["ValidationPerplexityDataset"]

LARGE_POSITIVE = int(1e10)


class ValidationPerplexityDataset(Dataset):
    r"""The dataset for validation perplexity."""

    instruction = ""
    evaluation_type = "perplexity"
    example_set = None
    extra_model_args = dict(max_tokens=0, temperature=0)

    @cached_property
    def references(self):
        return [instance["text"] for instance in self.evaluation_data]

    def post_processing(self, predictions: List[Tuple[float, int]]) -> List[float]:
        return np.array([result / length if length > 0 else LARGE_POSITIVE for result, length in predictions])
