import random
import re
from logging import getLogger
from typing import Any, List, Tuple, Union

import numpy as np

from ..metric import Accuracy
from .dataset import Dataset

logger = getLogger(__name__)

LARGE_POSITIVE = int(1e10)


class MultipleChoiceDataset(Dataset):
    r"""The dataset for multiple choice tasks. It ranks given options and is evaluated using `accuracy` score."""

    evaluation_type = "ranking"
    metrics = [Accuracy()]

    @staticmethod
    def shuffle_options(*values: List[Any]) -> List[int]:
        assert all(len(values[0]) == len(value) for value in values)
        order = list(range(len(values[0])))
        random.shuffle(order)

        for v in values:
            v[:] = [v[i] for i in order]

        return order

    @property
    def ranking_with_options(self):
        return not self.ranking_type.endswith("no_option")

    @ranking_with_options.setter
    def ranking_with_options(self, value: bool):
        if value:
            # remove suffix
            if self.ranking_type.endswith("no_option"):
                self.ranking_type = self.ranking_type[-len("no_option"):]
        else:
            self.ranking_type = "ppl_no_option"

    def _post_processing_ppl(self, predictions: np.ndarray) -> List[int]:
        labels = []
        st = 0

        for num in self.option_nums:
            if num <= 0:
                labels.append(-1)
                logger.warning(
                    f"Empty options detected in {self.display_name}. Please contact the author of the dataset."
                )
            else:
                labels.append(predictions[st:st + num].argmin())
            st += num
        return labels

    def _post_processing_prob(self, predictions: List[List[int]]) -> List[int]:
        labels = []
        for logit, option_num in zip(predictions, self.option_nums):
            if option_num <= 0:
                labels.append(-1)
                logger.warning(
                    f"Empty options detected in {self.display_name}. Please contact the author of the dataset."
                )
            elif logit is None:
                labels.append(-1)
            else:
                labels.append(np.argmax(logit).item() % option_num)
        return labels

    def _post_processing_generation(self, predictions: List[str]) -> List[int]:
        labels = []
        max_option_num = max(self.option_nums)

        # matches option labels in the text
        matches = r"\b([A-{op}])\b|\b([A-{op}])[\u2E80-\u9FFF]|[\u2E80-\u9FFF]([A-{op}])\b|[\u2E80-\u9FFF]([A-{op}])[\u2E80-\u9FFF]"
        option_regex = [re.compile(matches.format(op=chr(ord("A") + i))) for i in range(max_option_num)]

        for text, option_num in zip(predictions, self.option_nums):
            label_found = option_regex[option_num - 1].findall(text.strip().split("\n")[0])
            if not label_found:
                labels.append(-1)
            else:
                final_label = ""
                for op in label_found[-1]:
                    final_label = final_label or op
                labels.append(ord(final_label) - ord("A"))
        return labels

    def post_processing(self, predictions: Union[List[Tuple[float, int]], List[List[int]], List[str]]) -> List[int]:
        if self.model_evaluation_method == "get_ppl":
            if self.use_normalization:
                for m in self.metrics:
                    if isinstance(m, Accuracy):
                        m.tag = "Norm"
                normalized_predictions = np.array([rc[0] - ra[0] for rc, ra in zip(predictions[::2], predictions[1::2])])
            else:
                normalized_predictions = np.array([
                    result / length if length > 0 else LARGE_POSITIVE for result, length in predictions
                ])
            return self._post_processing_ppl(normalized_predictions)
        elif self.model_evaluation_method == "get_prob":
            return self._post_processing_prob(predictions)
        elif self.model_evaluation_method == "generation":
            return self._post_processing_generation(predictions)
