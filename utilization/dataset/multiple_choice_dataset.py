from typing import List, Tuple, Union

import numpy as np

from ..metric import Accuracy
from .dataset import Dataset


class MultipleChoiceDataset(Dataset):
    r"""The dataset for multiple choice tasks. It ranks given options and is evaluated using `accuracy` score."""

    evaluation_type = "ranking"
    metrics = [Accuracy()]

    def post_processing(self, predictions: Union[List[Tuple[float, int]], List[List[int]]]) -> List[int]:
        if self.model_evaluation_method == "get_ppl":
            labels = []
            st = 0
            predictions = np.array([result / length for result, length in predictions])
            for num in self.option_nums:
                labels.append(predictions[st:st + num].argmin())
                st += num
            predictions = labels
            return predictions
        elif self.model_evaluation_method == "get_prob":
            labels = []
            for logit, option_num in zip(predictions, self.option_nums):
                labels.append(np.argmax(logit).item() % option_num)
            return labels
