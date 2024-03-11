from typing import List

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
        "Gaokao_bench_score": float
    """

    def __init__(self, multiref_strategy="none"):
        self.multiref_strategy = multiref_strategy

    @staticmethod
    def _calculate_score(reference, prediction):
        answer = list(prediction)
        refs = reference["answer"]
        task = reference["task"]
        score = reference["score"]
        if task == "multi_question_choice" or task == "five_out_of_seven":
            get_score = 0
            total_score = 0
            for idx in range(len(answer)):
                get_score += score if answer[idx] == refs[idx] else 0
                total_score += score
            return get_score / total_score
        elif task == "single_choice":
            return 1 if answer[0] == refs[0] else 0
        else:
            target = [_ for _ in refs[0]]
            if len(answer) == 0:
                return 0
            pred = answer[0]
            correct_count = 0
            for _ in pred:
                if _ in target:
                    correct_count += 1
                else:
                    return 0
            return 1 if correct_count == len(target) else 0.5

    def __call__(self, predictions: List[tuple], references: List[List[dict]]):
        score_list = []
        for prediction, reference in zip(predictions, references):
            scores = [self._calculate_score(ref, prediction) for ref in reference]
            score_list.append(multi_ref_aggregation(scores, self.multiref_strategy))
        self._last_score_lists = {'Scoring rate': score_list}
        return {'Scoring rate': np.mean(score_list) * 100}
        #return {'Gaokao_bench_score': np.mean(score_list) * 100}
