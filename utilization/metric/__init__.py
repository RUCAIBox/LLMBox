import importlib
from typing import TYPE_CHECKING, Any, Type

from .metric import Metric

__all__ = [
    "avg_metrics", "Accuracy", "Bleu", "F1", "Em", "GaokaoBenchMetric", "GPTEval", "IFEval", "PassAtK",
    "PerspectiveApi", "Rouge", "WordAccuracy"
]

from .metric_utils import avg_metrics

if TYPE_CHECKING:
    from .accuracy import Accuracy as _Accuracy
    from .bleu import Bleu as _Bleu
    from .em_f1 import F1 as _F1
    from .em_f1 import Em as _Em
    from .gaokao_bench_metric import GaokaoBenchMetric as _GaokaoBenchMetric
    from .gpteval import GPTEval as _GPTEval
    from .ifeval import IFEval as _IFEval
    from .pass_at_k import PassAtK as _PassAtK
    from .perplexity import PPL as _PPL
    from .perspective_api import PerspectiveApi as _PerspectiveApi
    from .rouge import Rouge as _Rouge
    from .word_accuracy import WordAccuracy as _WordAccuracy


def lazy_import(module, instance) -> Any:
    try:
        module = importlib.import_module(f".{module}", __package__)
        instance = getattr(module, instance)
    except (ImportError, ModuleNotFoundError) as e:
        error_msg = e.__class__.__name__ + ": " + str(e)

        class ErrorMetric(Metric):

            def __init__(self, *args, **kwargs):
                self.error = error_msg

            def __call__(self, *args, **kwargs):
                raise RuntimeError(self.error)

        instance = ErrorMetric
    return instance


Accuracy: Type["_Accuracy"] = lazy_import("accuracy", "Accuracy")
Bleu: Type["_Bleu"] = lazy_import("bleu", "Bleu")
F1: Type["_F1"] = lazy_import("em_f1", "F1")
Em: Type["_Em"] = lazy_import("em_f1", "Em")
GaokaoBenchMetric: Type["_GaokaoBenchMetric"] = lazy_import("gaokao_bench_metric", "GaokaoBenchMetric")
GPTEval: Type["_GPTEval"] = lazy_import("gpteval", "GPTEval")
IFEval: Type["_IFEval"] = lazy_import("ifeval", "IFEval")
PassAtK: Type["_PassAtK"] = lazy_import("pass_at_k", "PassAtK")
PerspectiveApi: Type["_PerspectiveApi"] = lazy_import("perspective_api", "PerspectiveApi")
PPL: Type["_PPL"] = lazy_import("perplexity", "PPL")
Rouge: Type["_Rouge"] = lazy_import("rouge", "Rouge")
WordAccuracy: Type["_WordAccuracy"] = lazy_import("word_accuracy", "WordAccuracy")
