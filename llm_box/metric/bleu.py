import sacrebleu
import numpy as np
from .metric import Metric

class Bleu(Metric):
    r"""calculate the bleu_score
    """
    
    @staticmethod
    def __call__(predictions, references):
        scores = []
        for prediction, reference in zip(predictions, references):
            bleu = sacrebleu.sentence_bleu(prediction, [reference])
            scores.append(bleu.score)
        return {'Average SacreBleu sentences score': np.mean(scores)}