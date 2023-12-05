from .metric import Metric
import numpy as np


class Accuracy(Metric):
    r""" Calculate the Accuracy score.
    """

    def __init__(self, answer_type='str'):
        self.answer_type = answer_type

    def __call__(self, predictions, references):
        if self.answer_type == 'str':
            score_list = np.asarray(predictions) == np.asarray(references)
        elif self.answer_type == 'float':
            predictions = [float(p) if p.replace('.', '', 1).isdigit() else 0.0 for p in predictions]
            references = [float(r) for r in references]
            score_list = np.isclose(predictions, references)
        return {'Accuracy': np.mean(score_list) * 100}