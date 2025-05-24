import numpy as np

from .metric import Metric


class PPL(Metric):
    r"""Expose the ppl score.

    Return:
        "PPL": float
    """
    def __call__(self, predictions, _):
        self.last_score_lists = {"PPL": predictions}
        return {"PPL": np.mean(predictions)}
