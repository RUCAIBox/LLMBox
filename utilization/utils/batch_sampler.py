from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Tuple

from torch.utils.data.sampler import Sampler

if TYPE_CHECKING:
    from ..dataset.dataset import Dataset, DatasetCollection

logger = getLogger(__name__)


def info_dataset_group(dataset_group: List["Dataset"], group_length: int, model_attr: Any, model_kwargs: Any):
    subset_names = [d.subset_name for d in dataset_group if d.subset_name is not None]
    subset_names = (":" + ",".join(subset_names)) if len(subset_names) > 0 else ""
    instances = 0
    for d in dataset_group:
        instances += d.len(False, False, False)
        logger.debug(d)
    kwargs_name = d.model_evaluation_method.split("_")[-1] + "_kwargs"
    logger.info(
        f"Evaluating {d.model_evaluation_method} on {d.name}{subset_names} (model_attr={model_attr}, {kwargs_name}={model_kwargs}, len={group_length}, num_instances={instances})"
    )


def sample_dataset(total: int, batch_size: int) -> Iterator[List[int]]:
    for i in range(0, total, batch_size):
        yield list(range(i, min(i + batch_size, total)))


class DatasetCollectionBatchSampler(Sampler[List[int]]):

    def __init__(self, dataset_collection: "DatasetCollection", batch_size: int, vllm: bool = False):
        self.dataset_collection = dataset_collection
        self.batch_size = batch_size
        self.vllm = vllm
        self._forward_call = lambda *a, **k: RuntimeError("Not in dataset iteration context")
        self._splitted = self._split(self.dataset_collection)

    @staticmethod
    def _split(
        dataset_collection: "DatasetCollection"
    ) -> Tuple[List[int], List[Callable[[], None]], List[Callable[..., Any]]]:
        group_datasets: List[List["Dataset"]] = []
        group_lengths = []
        init_fns = []
        call_models = []
        last_hash = None
        model = dataset_collection._datasets[0].model
        for dataset in dataset_collection._datasets:
            cur_hash = (dataset._extra_model_args.items(), dataset.model_evaluation_method)
            if cur_hash != last_hash:

                def init_fn(group_idx: int):

                    def wrapper():
                        # use a callback function to index the entire group
                        kwargs = group_datasets[group_idx][0]._init_model()
                        info_dataset_group(
                            group_datasets[group_idx], group_lengths[group_idx], model._aggregate_model_attr(), kwargs
                        )

                    return wrapper

                group_lengths.append(0)
                group_datasets.append([])
                init_fns.append(init_fn(len(group_lengths) - 1))
                call_models.append(getattr(model, dataset.model_evaluation_method))

            group_lengths[-1] += dataset.len()
            group_datasets[-1].append(dataset)
            last_hash = cur_hash
        return group_lengths, init_fns, call_models

    def __iter__(self) -> Iterator[List[int]]:
        for total, init_model, self._forward_call in zip(*self._splitted):
            init_model()
            yield from sample_dataset(total, self.batch_size if not self.vllm else total)

    def call_model(self, *args, **kwargs) -> List[Any]:
        return self._forward_call(*args, **kwargs)  # type: ignore

    def __len__(self) -> int:
        return sum(dataset.len() // self.batch_size for dataset in self.dataset_collection._datasets)
