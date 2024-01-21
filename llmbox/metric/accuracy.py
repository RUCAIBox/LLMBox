import numpy as np

from .metric import Metric


class Accuracy(Metric):
    r"""Calculate the Accuracy score."""

    @staticmethod
    def __call__(predictions, references):
        score_list = np.asarray(predictions) == np.asarray(references)
        return {"Accuracy": np.mean(score_list) * 100}
