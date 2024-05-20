from typing import Optional

import numpy as np

from .metric import Metric


class Accuracy(Metric):
    r""" Calculate the Accuracy score.

    Return:
        "Accuracy": float
    """

    def __init__(self, tag: Optional[str] = None):
        self.tag = tag

    def __call__(self, predictions, references):
        score_list = np.asarray(predictions) == np.asarray(references)
        acc_tag = "Accuracy" if self.tag is None else "Accuracy:" + self.tag
        self.last_score_lists = {acc_tag: score_list}
        return {acc_tag: np.mean(score_list) * 100}
