from logging import getLogger
from typing import List, Tuple, Union

import numpy as np

from ..metric import Accuracy
from .dataset import Dataset

logger = getLogger(__name__)

LARGE_POSITIVE = 1e10


class MultipleChoiceDataset(Dataset):
    r"""The dataset for multiple choice tasks. It ranks given options and is evaluated using `accuracy` score."""

    evaluation_type = "ranking"
    metrics = [Accuracy()]

    def post_processing(self, predictions: Union[List[Tuple[float, int]], List[List[int]]]) -> List[int]:
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
