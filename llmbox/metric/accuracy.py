import numpy as np

from .metric import Metric


class Accuracy(Metric):
    r""" Calculate the Accuracy score.

    Return:
        "Accuracy": float
    """

    def __call__(self, predictions, references):
        score_list = np.asarray(predictions) == np.asarray(references)
        self._last_score_lists = {'Accuracy': score_list}
        return {'Accuracy': np.mean(score_list) * 100}
