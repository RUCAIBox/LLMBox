from typing import Set, Dict, Any



class Metric:
    r"""The base class object for all datasets."""

    def __init__(self):
        self.data: Dict[str, Any] = {}

    def add_batch(self, **kwargs):
        self._check_required_fields(**kwargs)
        for k, v in kwargs.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)

    @property
    def required_fields(self) -> Set[str]:
        raise NotImplementedError()

    def _check_required_fields(self, **kwargs):
        pass

    def calculate_metric(self, **kwargs) -> Dict[str, float]:
        raise NotImplementedError()

    def __add__(self, other):
        pass

