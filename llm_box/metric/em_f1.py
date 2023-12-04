from .metric import Metric
import numpy as np
import re
import string
from collections import Counter
from nltk import word_tokenize


def normalize_answer(s):
    """Lower text and remove punctuation, stories and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def multi_ref_aggregation(scores, multiref_strategy):
    if len(scores) > 1 and multiref_strategy == 'leave_one_out':
        func = lambda x: (max(x) * (len(x) - 1) + np.partition(x, -2)[-2]) / len(x)
    else:
        func = max
    return func(scores)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class Em(Metric):
    r""" Calculate the Exact Match score.
    """

    def __init__(self, multiref_strategy='none'):
        self.multiref_strategy = multiref_strategy

    @staticmethod
    def _calculate_em_score(reference, prediction):
        return int(normalize_answer(reference) == normalize_answer(prediction))

    def __call__(self, predictions, references):
        score_list = []
        for prediction, reference in zip(predictions, references):
            scores = [self._calculate_em_score(ref, prediction) for ref in reference]
            score_list.append(multi_ref_aggregation(scores, self.multiref_strategy))
        return {'EM': np.mean(score_list) * 100}


class F1(Metric):
    r""" Calculate the F1 score.
    """

    def __init__(self, multiref_strategy='none', force_number_match=False):
        self.multiref_strategy = multiref_strategy
        self.force_number_match = force_number_match

    @staticmethod
    def _calculate_f1_score(reference, prediction):
        ref_toks = word_tokenize(normalize_answer(reference))
        pred_toks = word_tokenize(normalize_answer(prediction))

        ref_num_toks = set([tok for tok in ref_toks if is_number(tok)])
        pred_num_toks = set([tok for tok in pred_toks if is_number(tok)])
        if ref_num_toks and not ref_num_toks.intersection(pred_num_toks):
            return 0

        common = Counter(ref_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(ref_toks) == 0 or len(pred_toks) == 0:
            return int(ref_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(ref_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def __call__(self, predictions, references):
        score_list = []
        for prediction, reference in zip(predictions, references):
            scores = [self._calculate_f1_score(ref, prediction) for ref in reference]
            score_list.append(multi_ref_aggregation(scores, self.multiref_strategy))
        return {'F1': np.mean(score_list) * 100}