import numpy as np

from .metric import Metric


class Accuracy(Metric):
    r""" Calculate the Accuracy score.

    Return:
        "Accuracy": float
    """

    def __call__(self, predictions, references):
        if isinstance(references, dict) and references['dataset'] == "winogender":
            def cal_sub_score(predictions, references, i):
                indices = [i for i, value in enumerate(references["gender"]) if value == i]
                predictions = [predictions[i] for i in indices]
                references = [references["references"][i] for i in indices]
                score_list = np.asarray(predictions) == np.asarray(references)
                return score_list
            gender_dict = {"neutral": 0, "male": 1, "female": 2}
            score = {}
            for key in gender_dict:
                score_list = cal_sub_score(predictions, references, gender_dict[key])
                self._last_score_lists.update({f"Accuracy_{key}": score_list})
                score.update({f"Accuracy_{key}": np.mean(score_list) * 100})
            score_list = np.asarray(predictions) == np.asarray(references)
            self._last_score_lists.update({'Accuracy_all': score_list})
            score.update({'Accuracy_all': np.mean(score_list) * 100})
            return score

        score_list = np.asarray(predictions) == np.asarray(references)
        self._last_score_lists = {'Accuracy': score_list}
        return {'Accuracy': np.mean(score_list) * 100}
