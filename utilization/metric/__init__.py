import importlib
from typing import Type

from .metric import Metric

__all__ = [
    "avg_metrics", "Accuracy", "Bleu", "F1", "Em", "Gaokao_bench_metric", "GPTEval", "IFEval", "PassAtK",
    "Perspective_api", "Rouge", "Word_Accuracy"
]

from .utils import avg_metrics


def lazy_import(module, instance) -> Type[Metric]:
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


Accuracy = lazy_import("accuracy", "Accuracy")
Bleu = lazy_import("bleu", "Bleu")
F1 = lazy_import("em_f1", "F1")
Em = lazy_import("em_f1", "Em")
Gaokao_bench_metric = lazy_import("gaokao_bench_metric", "Gaokao_bench_metric")
GPTEval = lazy_import("gpteval", "GPTEval")
IFEval = lazy_import("ifeval", "IFEval")
PassAtK = lazy_import("pass_at_k", "PassAtK")
Perspective_api = lazy_import("perspective_api", "Perspective_api")
Rouge = lazy_import("rouge", "Rouge")
Word_Accuracy = lazy_import("word_accuracy", "Word_Accuracy")
