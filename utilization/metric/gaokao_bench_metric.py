from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .metric import Metric


def multi_ref_aggregation(scores, multiref_strategy):
    if len(scores) > 1 and multiref_strategy == "leave_one_out":
        func = lambda x: (max(x) * (len(x) - 1) + np.partition(x, -2)[-2]) / len(x)
    else:
        func = max
    return func(scores)


class Gaokao_bench_metric(Metric):
    r""" Calculate the Gaokao-Bench score.

    Return:
        "Scoring rate": float
    """

    def __init__(self, multiref_strategy="none"):
        self.multiref_strategy = multiref_strategy

    @staticmethod
    def _calculate_score(prediction, reference):
        answer = list(prediction)
        refs = reference["answer"]
        task = reference["task"]
        score = reference["score"]
        if len(answer) == 0:
            return 0
        if task == "multi_mcqs" or task == "seven_option":
            total_score = 0
            for idx in range(len(answer)):
                total_score += score if answer[idx] == refs[idx] else 0
            return total_score
        elif task == "single_answer_mcq":
            return score if answer[0] == refs[0] else 0
        else:
            target = [_ for _ in refs[0]]
            pred = answer[0]
            correct_count = 0
            for _ in pred:
                if _ in target:
                    correct_count += 1
                else:
                    return 0
            return score if correct_count == len(target) else score / 2

    def __call__(self, predictions: List[tuple], references: List[List[dict]]):
        score_list = []
        total_score = 0
        for prediction, reference in zip(predictions, references):
            total_score += reference[0]["score"] * len(reference[0]["answer"])
            scores = [self._calculate_score(prediction, ref) for ref in reference]
            score_list.append(multi_ref_aggregation(scores, self.multiref_strategy))
        self.last_score_lists = {'Scoring rate': score_list}
        return {'Scoring rate': np.sum(score_list) / total_score * 100}
