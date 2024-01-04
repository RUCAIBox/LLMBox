import torch
import ignite.metrics
from .metric import Metric


class Rouge(Metric):
    r"""Calculate the ROUGE score, including ROUGE_1, ROUGE_2, ROUGE_L
    """

    @staticmethod
    def __call__(summaries, references):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        score_calc = ignite.metrics.Rouge(variants=["L", 1, 2], device=device)
        for summary, reference in zip(summaries, references):
            score_calc.update(([summary.split()], [[reference.split()]]))
        list_of_score = score_calc.compute()
        return {"ROUGE_1": list_of_score["Rouge-1-F"], "ROUGE_2": list_of_score["Rouge-2-F"], "ROUGE_L": list_of_score["Rouge-L-F"]}
