from typing import Set, Dict
from .metric import Metric
import numpy as np


class Accuracy(Metric):

    @property
    def required_fields(self) -> Set[str]:
        return {'references', 'predictions'}

    def calculate_metric(self, predictions, references) -> Dict[str, float]:
        score_list = np.asarray(predictions) == np.asarray(references)
        return {'accuracy': np.mean(score_list)}
