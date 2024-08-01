from itertools import chain, islice
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Tuple

from torch.utils.data.sampler import Sampler

from .prefix_caching import CachePrefixSampler, round_down

if TYPE_CHECKING:
    from ...dataset.dataset import Dataset, DatasetCollection

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
    real_shots = [d.real_num_shots for d in dataset_group if d.real_num_shots is not None]
    if real_shots:
        min_num_shots = min(real_shots)
        max_num_shots = max(real_shots)
        if min_num_shots != max_num_shots:
            num_shots = f"{min_num_shots}-{max_num_shots}"
        else:
            num_shots = str(min_num_shots)
    else:
        num_shots = "None"
    model_evaluation_method = dataset_group[0].model_evaluation_method
    if model_evaluation_method == "get_ppl":
        model_evaluation_method += f" ({dataset_group[0].ranking_type})"
    logger.info(
        f"Evaluating {model_evaluation_method} on {d.display_name}{subset_str} (model_attr={model_attr}, {kwargs_name}={model_kwargs}, num_shots={num_shots}, len={group_length}, num_instances={instances}, use_cache={use_cache})"
    )
    logger.debug(f"Datasets: {d.dataset_name}{subset_names}")


class AutoBatchSizeSampler(Sampler[List[int]]):

    def __init__(self, data, batch_size: int, auto_batch_size: bool, index_offset: int = 0):
        """Sampler that automatically adjusts the batch size based on the maximum length of the data.

        Args:
            data: The data to sample from.
            batch_size: The maximum batch size.
            auto_batch_size: Whether to automatically adjust the batch size based on the maximum length of the data.
            index_offset: The  offset of indices to yield.
        """
        self.data = [src.to_model_prompt() if hasattr(src, "to_model_prompt") else src for src in data]
        total = len(self.data)
        self.batch_size = batch_size
        self.auto_batch_size = auto_batch_size
        self.first_max_len = None
        self.data_order = [[]]
        self.index_offset = index_offset
        """The data indices to yield (batches of indices). In convenience of the `__iter__` method, the indices are offset-based: `range(index_offset, index_offset + total)`."""
        logger.debug(self)

        if not self.auto_batch_size:
            for i in range(0, total, self.batch_size):
                st = i + self.index_offset
                ed = min(i + self.batch_size, total) + self.index_offset
                self.data_order[-1].extend(range(st, ed))
                if len(self.data_order[-1]) == self.batch_size:
                    self.data_order.append([])
        else:
            for i in range(total):
                self.data_order[-1].append(i + self.index_offset)
                if self.check_new_batch(self.data_order[-1], i + 1):
                    self.data_order.append([])

        # remove the last empty batches
        while self.data_order[-1] == []:
            self.data_order.pop()
        logger.debug(f"AutoBatchSizeSampler: {len(self.data_order)} batches starting from {self.index_offset}")

    def check_new_batch(self, offset_query_indices: List[int], next_data: int) -> bool:
        """Check the condition to start a new batch."""

        current_batch = len(offset_query_indices)
        if not self.auto_batch_size:
            return current_batch > self.batch_size

        # data: 0-based
        # offset_query_indices: offset-based
        # next_data: 0-based
        max_len = max(len(self.data[q - self.index_offset]) for q in offset_query_indices)
        if next_data < len(self.data):
            max_len = max(len(self.data[next_data]), max_len)

        if self.first_max_len is None:
            self.first_max_len = max_len

        available_space = self.batch_size * self.first_max_len

        batch_size = available_space // max_len
        batch_size = round_down(batch_size)
        return current_batch >= batch_size

    def __iter__(self) -> Iterator[List[int]]:
        yield from self.data_order

    def __repr__(self) -> str:
        return f"AutoBatchSizeSampler(batch_size={self.batch_size}, auto_batch_size={self.auto_batch_size}, index_offset={self.index_offset})"


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
        skip = dataset_collection.args.continue_from
        for dataset in dataset_collection._datasets:
            if dataset.len() <= skip:
                skip -= dataset.len()
                continue
            # check if the model arguments has changed
            cur_hash = (dataset._extra_model_args.items(), dataset.model_evaluation_method, dataset.total_prefix_num)
            if cur_hash != last_hash:

                def init_fn(group_idx: int):

                    def wrapper():
                        # use a callback function to index the entire group
                        kwargs = group_datasets[group_idx][0]._init_model()
                        use_cache = all(d.hf_prefix_caching for d in group_datasets[group_idx])
                        total_prefix_num = group_datasets[group_idx][0].total_prefix_num if use_cache else 0
                        info_dataset_group(
                            group_datasets[group_idx], group_lengths[group_idx], model._aggregate_model_attr(), kwargs,
                            use_cache
                        )
                        iterator = chain.from_iterable(group_datasets[group_idx])
                        original_len = sum([d.len() for d in group_datasets[group_idx]])
                        if group_lengths[group_idx] != original_len:
                            iterator = islice(iterator, original_len - group_lengths[group_idx], None)
                        return iterator, total_prefix_num

                    return wrapper

                group_lengths.append(0)
                group_datasets.append([])
                init_fns.append(init_fn(len(group_lengths) - 1))
                call_models.append(getattr(model, dataset.model_evaluation_method))

            if skip:
                group_lengths[-1] += dataset.len() - skip
                skip = 0
            else:
                group_lengths[-1] += dataset.len()
            group_datasets[-1].append(dataset)
            last_hash = cur_hash
        return group_lengths, init_fns, call_models

    def __iter__(self) -> Iterator[List[int]]:
        model = self.dataset_collection._datasets[0].model
        accumulative_offset = 0

        # iterate over the dataset groups
        for group_total, init_model, self._forward_call in zip(*self._splitted):
            iterator, total_prefix_num = init_model()
            logger.debug("New sub-collection, total iteration: %d", group_total)
            if total_prefix_num > 1 and model.support_cache:
                sampler = CachePrefixSampler(
                    data=iterator,
                    total=group_total,
                    total_prefix_num=total_prefix_num,
                    batch_size=self.batch_size,
                    auto_batch_size=self.auto_batch_size,
                    index_offset=accumulative_offset,
                )
                model.set_cacher(sampler)
                yield from sampler
            else:
                # disable prefix_caching
                model.use_cache = False
                # dynamic batch size for vLLM
                yield from AutoBatchSizeSampler(
                    iterator,
                    self.batch_size if not self.vllm else group_total,
                    self.auto_batch_size and not self.vllm,
                    index_offset=accumulative_offset
                )
            accumulative_offset += group_total

    def call_model(self, *args, **kwargs) -> List[Any]:
        """Route the model to call the corresponding `model_evaluation_method`"""
        return self._forward_call(*args, **kwargs)  # type: ignore

    def __len__(self) -> int:
        return sum(dataset.len() // self.batch_size for dataset in self.dataset_collection._datasets)
