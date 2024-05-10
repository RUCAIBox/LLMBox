import re
from logging import getLogger
from typing import List, Tuple, Union

import numpy as np

from ..metric import Accuracy
from .dataset import Dataset

logger = getLogger(__name__)

LARGE_POSITIVE = int(1e10)


class MultipleChoiceDataset(Dataset):
    r"""The dataset for multiple choice tasks. It ranks given options and is evaluated using `accuracy` score."""

    evaluation_type = "ranking"
    metrics = [Accuracy()]

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

    def post_processing(self, predictions: Union[List[Tuple[float, int]], List[List[int]], List[str]]) -> List[int]:
        if self.model_evaluation_method == "get_ppl":
            labels = []
            st = 0
            predictions = np.array([
                result / length if length > 0 else LARGE_POSITIVE for result, length in predictions
            ])
            for num in self.option_nums:
                if num <= 0:
                    labels.append(-1)
                    logger.warning(
                        f"Empty options detected in {self.dataset_name}. Please contact the author of the dataset."
                    )
                else:
                    labels.append(predictions[st:st + num].argmin())
                st += num
            predictions = labels
            return predictions
        elif self.model_evaluation_method == "get_prob":
            labels = []
            for logit, option_num in zip(predictions, self.option_nums):
                if option_num <= 0:
                    labels.append(-1)
                    logger.warning(
                        f"Empty options detected in {self.dataset_name}. Please contact the author of the dataset."
                    )
                elif logit is None:
                    labels.append(-1)
                else:
                    labels.append(np.argmax(logit).item() % option_num)
            return labels
        elif self.model_evaluation_method == "generation":
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
