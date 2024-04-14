from itertools import chain
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Tuple

from torch.utils.data.sampler import Sampler

from .prefix_caching import CachePrefixSampler

if TYPE_CHECKING:
    from ..dataset.dataset import Dataset, DatasetCollection

logger = getLogger(__name__)


def info_dataset_group(
    dataset_group: List["Dataset"], group_length: int, model_attr: Any, model_kwargs: Any, use_cache: bool
):
    subset_names = [d.subset_name for d in dataset_group if d.subset_name is not None]
    subset_str = (f" {len(subset_names)} subsets") if len(subset_names) > 0 else ""
    instances = 0
    for d in dataset_group:
        instances += d.len(False, False, False)
        logger.debug(d)
    kwargs_name = d.model_evaluation_method.split("_")[-1] + "_kwargs"
    logger.info(
        f"Evaluating {d.model_evaluation_method} on {d.name}{subset_str} (model_attr={model_attr}, {kwargs_name}={model_kwargs}, len={group_length}, num_instances={instances}, use_cache={use_cache})"
    )
    logger.debug(f"Datasets: {d.name}{subset_names}")


def sample_dataset(total: int, batch_size: int) -> Iterator[List[int]]:
    for i in range(0, total, batch_size):
        yield list(range(i, min(i + batch_size, total)))


class DatasetCollectionBatchSampler(Sampler[List[int]]):

    def __init__(
        self,
        dataset_collection: "DatasetCollection",
        batch_size: int,
        vllm: bool = False,
        auto_batch_size: bool = False,
    ):
        self.dataset_collection = dataset_collection
        self.batch_size = batch_size
        self.vllm = vllm
        self.auto_batch_size = auto_batch_size
        self._forward_call = lambda *a, **k: RuntimeError("Not in dataset iteration context")
        self._splitted = self._split(self.dataset_collection)

    @staticmethod
    def _split(
        dataset_collection: "DatasetCollection"
    ) -> Tuple[List[int], List[Callable[[], Tuple[Iterator, int]]], List[Callable[..., Any]]]:
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
                        use_cache = all(d.prefix_caching for d in group_datasets[group_idx])
                        total_prefix_num = group_datasets[group_idx][0].total_prefix_num if use_cache else 0
                        info_dataset_group(
                            group_datasets[group_idx], group_lengths[group_idx], model._aggregate_model_attr(), kwargs,
                            use_cache
                        )
                        iterator = chain.from_iterable(group_datasets[group_idx])
                        return iterator, total_prefix_num

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
        model = self.dataset_collection._datasets[0].model
        for total, init_model, self._forward_call in zip(*self._splitted):
            iterator, total_prefix_num = init_model()
            if total_prefix_num > 1:
                sampler = CachePrefixSampler(iterator, total, total_prefix_num, self.batch_size, self.auto_batch_size)
                model.set_cacher(sampler)
                yield from sampler
            else:
                # disaable prefix_caching
                model.use_cache = False
                # dynamic batch size for vLLM
                yield from sample_dataset(total, self.batch_size if not self.vllm else total)

    def call_model(self, *args, **kwargs) -> List[Any]:
        return self._forward_call(*args, **kwargs)  # type: ignore

    def __len__(self) -> int:
        return sum(dataset.len() // self.batch_size for dataset in self.dataset_collection._datasets)
