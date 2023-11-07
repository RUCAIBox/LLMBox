import numpy as np
from .dataset import Dataset


class MultipleChoiceDataset(Dataset):
    r"""The dataset for multiple choice tasks. It ranks given options and is evaluated using `accuracy` score.
    """

    evaluation_type = "ranking"
    metric = "accuracy"

    def __init__(self, args, model):
        super().__init__(args, model)

    def calculate_metric(self, predictions):
        score_list = np.asarray(predictions) == np.asarray(self.references)
        return {'Accuracy': np.mean(score_list)}
