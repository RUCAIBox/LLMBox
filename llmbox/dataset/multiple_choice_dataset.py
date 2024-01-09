import numpy as np

from .dataset import Dataset
from ..metric import Accuracy


class MultipleChoiceDataset(Dataset):
    r"""The dataset for multiple choice tasks. It ranks given options and is evaluated using `accuracy` score.
    """

    evaluation_type = "ranking"
    metrics = [Accuracy()]

    def post_processing(self, predictions):
        labels = []
        st = 0
        predictions = np.array([result / length for result, length in predictions])
        for num in self.option_nums:
            labels.append(predictions[st:st + num].argmin())
            st += num
        predictions = labels

        return predictions
