import numpy as np
from .dataset import Dataset

class GenerationDataset(Dataset):
    r"""The dataset for Generation probles. It sloves problems in nature language and is evaluated using `accuracy` score.
    """

    evaluation_type = "generation"
    metric = "accuracy"

    def __init__(self, args, model):
        super().__init__(args, model)

    def calculate_metric(self, predictions):
        predictions = [self.answer_cleansing(prediction) for prediction in predictions]
        score_list = np.asarray(predictions) == np.asarray(self.references)
        return {'Accuracy': np.mean(score_list)}