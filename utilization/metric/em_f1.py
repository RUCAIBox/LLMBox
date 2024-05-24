import re
import string
from collections import Counter
from typing import Callable, List, Literal, Tuple, Union

import nltk
import numpy as np

from .drop_utils import process_results
from .metric import Metric

_ARTICLES = re.compile(r"\b(a|an|the|and)\b", re.UNICODE)
_TOKENIZE_SEP = re.compile(r" |-")

_TOKENIZER_DICT = {
    "nltk": nltk.word_tokenize,
    "split": str.split,
    "regex": _TOKENIZE_SEP.split,
}


def normalize_answer(s: Union[str, List[str], Tuple[str]]) -> str:
    """Lower text and remove punctuation, stories and extra whitespace."""

    def remove_articles(text):
        return _ARTICLES.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if isinstance(s, (tuple, list)):
        s = " ".join(s)

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def multi_ref_aggregation(scores, multiref_strategy: Literal["max", "leave_one_out"]):
    if len(scores) > 1 and multiref_strategy == "leave_one_out":
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

    Return:
        "EM": float
    """

    def __init__(
        self,
        *,
        dataset: Literal["independent"] = "independent",
        multiref_strategy: Literal["max", "leave_one_out"] = "max",
    ):
        self.dataset = dataset
        self.multiref_strategy = multiref_strategy

    @staticmethod
    def _calculate_em_score(
        reference: Union[str, List[str]],
        prediction: str,
    ):
        """Calculate EM score for a single example."""
        return int(normalize_answer(reference) == normalize_answer(prediction))

    def get_metrics(self, pred: str, refs: Union[str, List[str], Tuple[str]]) -> float:
        scores = [self._calculate_em_score(ref, pred) for ref in refs]
        return multi_ref_aggregation(scores, self.multiref_strategy) * 100

    def get_metrics_drop(self, pred: str, refs: List[str]) -> float:
        return process_results(pred=pred, golds=refs)["EM"] * 100

    def __call__(self, predictions: List[str], references: List[Union[str, List[str], Tuple[str]]]):
        score_list = []
        if self.dataset == "independent":
            get_metrics = self.get_metrics

        for prediction, reference in zip(predictions, references):
            score_list.append(get_metrics(prediction, reference))

        self.last_score_lists = {'EM': score_list}
        return {'EM': np.mean(score_list)}


class F1(Metric):
    r""" Calculate the F1 score.

    Args:
        `multiref_strategy`: Strategy to aggregate F1 scores for multiple references.
        `force_number_match`: If reference contains numbers, prediction must matches all the numbers in the reference.
        `word_tokenize`: Tokenizer functions for different tokenization methods. Default: nltk.word_tokenize.
            DROP: https://github.com/EleutherAI/lm-evaluation-harness/blob/3196e907fa195b684470a913c7235ed7f08a4383/lm_eval/tasks/drop/utils.py#L193
            SQuAD: https://github.com/huggingface/datasets/blob/f96e74d5c633cd5435dd526adb4a74631eb05c43/metrics/squad_v2/evaluate.py#L80

    Return:
        "F1": float
    """

    def __init__(
        self,
        *,
        dataset: Literal["independent"] = "independent",
        multiref_strategy: Literal["max", "leave_one_out"] = "max",
        word_tokenize: Literal["nltk", "split", "regex"] = "nltk",
        normalize_level: Literal["token", "text", "both"] = "both",
        align_bag: Literal["counter", "set"] = "counter",
        force_number_match=False,
    ):
        self.dataset = dataset
        self.word_tokenize = _TOKENIZER_DICT[word_tokenize]
        self.normalize_level = normalize_level
        self.multiref_strategy = multiref_strategy
        self.align_bag = align_bag
        self.force_number_match = force_number_match

    @staticmethod
    def _calculate_f1_score(
        *,
        reference: Union[str, List[str], Tuple[str]],
        prediction: str,
        word_tokenize: Callable[[str], List[str]],
        normalize_token: bool = True,
        normalize_text: bool = True,
        align_bag_set: bool = False,
        force_number_match=False,
    ):
        """Calculate F1 score for a single example."""

        if isinstance(reference, (list, tuple)):
            reference = " ".join(reference)

        # normalize -> tokenize https://github.com/huggingface/datasets/blob/f96e74d5c633cd5435dd526adb4a74631eb05c43/metrics/squad_v2/evaluate.py#L80
        # tokenize -> normalize https://github.com/EleutherAI/lm-evaluation-harness/blob/3196e907fa195b684470a913c7235ed7f08a4383/lm_eval/tasks/drop/utils.py#L197
        if normalize_text:
            reference = normalize_answer(reference)
            prediction = normalize_answer(prediction)

        if normalize_token:
            normalize_token_fn = normalize_answer
        else:
            normalize_token_fn = lambda x: x

        ref_toks = [normalize_token_fn(tok) for tok in word_tokenize(reference)]
        pred_toks = [normalize_token_fn(tok) for tok in word_tokenize(prediction)]

        if force_number_match and all(is_number(tok) for tok in ref_toks):
            ref_num_toks = set([tok for tok in ref_toks if is_number(tok)])
            pred_num_toks = set([tok for tok in pred_toks if is_number(tok)])
            if ref_num_toks and not ref_num_toks.intersection(pred_num_toks):
                return 0

        if align_bag_set:
            ref_toks = set(ref_toks)
            pred_toks = set(pred_toks)

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

    def get_metrics(self, pred: str, refs: Union[str, List[str], Tuple[str]]) -> float:
        scores = [
            self._calculate_f1_score(
                reference=ref,
                prediction=pred,
                normalize_token=self.normalize_level in ["token", "both"],
                normalize_text=self.normalize_level in ["text", "both"],
                align_bag_set=self.align_bag == "set",
                word_tokenize=self.word_tokenize,
                force_number_match=self.force_number_match,
            ) for ref in refs
        ]
        return multi_ref_aggregation(scores, self.multiref_strategy) * 100

    def __call__(self, predictions: List[str], references: List[Union[str, List[str], Tuple[str]]]):
        score_list = []
        if self.dataset == "independent":
            get_metrics = self.get_metrics

        for prediction, reference in zip(predictions, references):
            f1 = get_metrics(prediction, reference)
            score_list.append(f1)

        self.last_score_lists = {'F1': score_list}
        return {"F1": np.mean(score_list)}
