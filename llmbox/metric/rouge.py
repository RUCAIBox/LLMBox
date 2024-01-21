import numpy as np
from rouge_score import rouge_scorer

from .metric import Metric


class Rouge(Metric):
    r"""Calculate the ROUGE score, including ROUGE_1, ROUGE_2, ROUGE_L

    Return:
        "ROUGE_1": float
        "ROUGE_2": float
        "ROUGE_L": float
    """

    def __call__(self, predictions, references):
        score_rouge1 = []
        score_rouge2 = []
        score_rougeL = []
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        for prediction, reference in zip(predictions, references):
            scores = scorer.score(prediction, reference)
            score_rouge1.append(scores["rouge1"].fmeasure)
            score_rouge2.append(scores["rouge2"].fmeasure)
            score_rougeL.append(scores["rougeL"].fmeasure)
        self._last_score_lists = {"ROUGE-1": score_rouge1, "ROUGE-2": score_rouge2, "ROUGE_L": score_rougeL}
        return {
            "ROUGE-1": np.mean(score_rouge1) * 100,
            "ROUGE-2": np.mean(score_rouge2) * 100,
            "ROUGE_L": np.mean(score_rougeL) * 100,
        }
