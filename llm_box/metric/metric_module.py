from .metric import Metric
from typing import Dict, Iterable, Optional, Set, Union
from .utils import load_metric


class MetricModule(Metric):
    r"""A MetricModule is a collection of metrics."""

    def __init__(self, metrics: Optional[Union[str, Iterable[str]]] = None):
        super().__init__()
        if metrics is None:
            self.metrics: Dict[str, Metric] = {}
        else:
            if isinstance(metrics, str):
                metrics = [metrics]

            self.metrics = {m: load_metric(m) for m in metrics}

    @property
    def required_fields(self) -> Set[str]:
        required_fields = set()
        for metric in self.metrics.values():
            required_fields = required_fields.union(metric.required_fields)
        return required_fields

    def calculate_metric(self) -> Dict[str, float]:
        all_results = {}
        for key, metric in self.metrics.items():
            required_data = {k: v for k, v in self.data.items() if k in metric.required_fields}
            metric_results = metric.calculate_metric(**required_data)
            all_results.update(metric_results)
        return all_results
