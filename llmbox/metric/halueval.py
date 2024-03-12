import numpy as np

from .metric import Metric


class HaluEval(Metric):
    r""" Calculate the HaluEval score.

    Return:
        "Accuracy": float
    """

    def __init__(self, type="qa"):
        self.type = type

    def __call__(self, predictions, references):
        score_list = []
        for pred, refer in zip(predictions, references):
            if ("Yes" in pred and "No" in pred) or ("Yes" not in pred and "No" not in pred):
                score_list.append(0)
                continue
            elif "Yes" in pred:
                if pred != "Yes":
                    pred = "Yes"
            elif "No" in pred:
                if pred != "No":
                    pred = "No"
            else:
                score_list.append(0)

            if pred == refer:
                score_list.append(1)
            else:
                score_list.append(0)
        score_list = np.array(score_list)
        self._last_score_lists = {f'Accuracy({self.type})': score_list}
        return {f'Accuracy({self.type})': np.mean(score_list) * 100}
