import sacrebleu
import numpy as np
from .metric import Metric


class Bleu(Metric):
    r"""Calculate the BLEU score.
    """

    @staticmethod
    def __call__(predictions, references):
        scores = []
        for prediction, reference in zip(predictions, references):
            sentence_bleu = sacrebleu.sentence_bleu(prediction, [reference], tokenize='intl')
            scores.append(sentence_bleu.score)
        corpus_bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize='intl')
        return {'Sentence_BLEU': np.mean(scores), 'Corpus_BLEU': corpus_bleu.score}
