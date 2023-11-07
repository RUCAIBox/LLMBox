from ..utils import import_main_class
from .metric import Metric as LLMMetric


def load_metric(metric: str, *args, **kwargs):
    r"""Load corresponding metric class.

    Args:
        metric (str): The name of metric.

    Returns:
        Metric: Our class for metric.
    """
    metric_cls = import_main_class('..' + metric, LLMMetric, package=__name__)
    metric = metric_cls(*args, **kwargs)
    return metric
