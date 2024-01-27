import numpy as np
import sacrebleu

from .metric import Metric


class Bleu(Metric):
    r"""Calculate the BLEU score.

    Return:
        "Sentence_BLEU": float
        "Corpus_BLEU": float
    """

    def __call__(self, predictions, references):
        scores = []
        for prediction, reference in zip(predictions, references):
            sentence_bleu = sacrebleu.sentence_bleu(prediction, [reference], tokenize="intl")
            scores.append(sentence_bleu.score)
        corpus_bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize='intl')
        self._last_score_lists = {'Sentence_BLEU': scores}
        return {'Sentence_BLEU': np.mean(scores), 'Corpus_BLEU': corpus_bleu.score}
