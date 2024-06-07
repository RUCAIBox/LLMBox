import importlib
from typing import TYPE_CHECKING, Any, Type

from .metric import Metric

__all__ = [
    "avg_metrics", "Accuracy", "Bleu", "F1", "Em", "Gaokao_bench_metric", "GPTEval", "IFEval", "PassAtK",
    "Perspective_api", "Rouge", "Word_Accuracy"
]

from .metric_utils import avg_metrics

if TYPE_CHECKING:
    from .accuracy import Accuracy as _Accuracy
    from .bleu import Bleu as _Bleu
    from .em_f1 import F1 as _F1
    from .em_f1 import Em as _Em
    from .gaokao_bench_metric import Gaokao_bench_metric as _Gaokao_bench_metric
    from .gpteval import GPTEval as _GPTEval
    from .ifeval import IFEval as _IFEval
    from .pass_at_k import PassAtK as _PassAtK
    from .perspective_api import Perspective_api as _Perspective_api
    from .rouge import Rouge as _Rouge
    from .word_accuracy import Word_Accuracy as _Word_Accuracy


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
Gaokao_bench_metric: Type["_Gaokao_bench_metric"] = lazy_import("gaokao_bench_metric", "Gaokao_bench_metric")
GPTEval: Type["_GPTEval"] = lazy_import("gpteval", "GPTEval")
IFEval: Type["_IFEval"] = lazy_import("ifeval", "IFEval")
PassAtK: Type["_PassAtK"] = lazy_import("pass_at_k", "PassAtK")
Perspective_api: Type["_Perspective_api"] = lazy_import("perspective_api", "Perspective_api")
Rouge: Type["_Rouge"] = lazy_import("rouge", "Rouge")
Word_Accuracy: Type["_Word_Accuracy"] = lazy_import("word_accuracy", "Word_Accuracy")
