import numpy as np
from .dataset import Dataset

class MathWordDataset(Dataset):
    r"""The dataset for math word probles. It sloves math word problems in nature language and is evaluated using `accuracy` score.
    """

    evaluation_type = "generation"
    metric = "accuracy"

    def __init__(self, args):
        super().__init__(args)

    def calculate_metric(self, predictions):
        score_list = np.asarray(predictions) == np.asarray(self.references)
        return {'Accuracy': np.mean(score_list)}