from logging import getLogger
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data.sampler import Sampler
from transformers import DynamicCache

from .conversation import Conversation

_LegacyCache = Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], ...]

logger = getLogger(__name__)


class SequenceCache(DynamicCache):
    """
    A cache specifically designed for storing sequence-level state in key-value
    pairs, with additional support for caching logits of the next predicted
    token. This facilitates sequence generation tasks by reusing computed
    states and reducing redundancy.

    The cache structure for keys and values is organized as follows:
    Shape: `[BatchSize, NumHeads, SeqLength, EmbedSizePerHead]`
    """

    def __init__(self) -> None:
        # keeps cache in a list instead of a stacked tensor because the tensor may on different devices
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.next_logits: List[torch.Tensor] = []  # used in `get_ppl` to concatenate logits
        self.last_texts: List[str] = []  # used in `get_cache` to concatenate tokens
        self.token_ids: Optional[torch.Tensor] = None
        self.real_seq_length: List[int] = []
        self.cache_level = None
        self.pad_token_id = 0

    def set_last_text(self, last_texts: Union[str, List[str]]):
        """Set the last part of sequence to provide appropriate context for
        tokenization in subsequent sequence processing."""
        if isinstance(last_texts, str):
            last_texts = [last_texts]

        if len(last_texts) != self.get_seq_num():
            raise ValueError(
                f"last_texts ({len(last_texts)}) should be a list of strings with the same length as the cache ({self.get_seq_num()})"
            )

        self.last_texts = last_texts

    def set_next_logits(self, last_logits: Union[torch.Tensor, List[torch.Tensor]]):
        """Save the logits for the next token to avoid recomputation in
        sequential generation tasks."""
        if isinstance(last_logits, torch.Tensor):
            last_logits = [last_logits]

        assert all(t.shape[0] == 1 and t.shape[1] == 1 for t in last_logits)

        if len(last_logits) != self.get_seq_num():
            raise ValueError(
                f"last_logits ({len(last_logits)}) should be a list of tensors with the same length as the cache ({self.get_seq_num()})"
            )
        self.next_logits = last_logits

    def set_token_ids(self, token_ids: torch.Tensor):
        """Associate the specified token IDs with the current cache entries,
        aligning them with the corresponding input sequences."""

        if token_ids.shape[0] != self.get_seq_num():
            raise ValueError(
                f"token_ids ({len(token_ids)}) should be a list of tensors with the same length as the cache ({self.get_seq_num()})"
            )

        self.token_ids = token_ids

    def get_token_ids(self, next_logits: bool = False, device: Optional[str] = None) -> Optional[torch.Tensor]:
        """Retrieve the token IDs from the cache, optionally appending the next
        token based on the cached logits.

        Args:
            with_next (bool): If True, append the next token (greedily decoded
            from `next_logits`) to the returned token IDs."""
        if self.token_ids is None:
            return None

        if next_logits:
            if len(self.next_logits) == 0:
                raise ValueError("No next logits available to append to token IDs")
            next_logits = torch.cat(self.next_logits, dim=0).to(device)
            if device is None:
                device = next_logits.device
            return torch.cat([self.token_ids.to(device), next_logits.argmax(dim=-1)], dim=1)
        else:
            return self.token_ids.to(device)

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[_LegacyCache] = None) -> "SequenceCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = SequenceCache()
        if past_key_values is not None:
            batch_size, _, seq_len, _ = past_key_values[0][0].shape
            for key_states, value_states in past_key_values:
                cache.key_cache.append(key_states.detach())
                cache.value_cache.append(value_states.detach())
            cache.real_seq_length = [seq_len] * batch_size
        return cache

    def get_seq_num(self) -> int:
        return len(self.real_seq_length)

    def trim(self, num_l: int = 0, num_r: int = 0):
        if num_l > 0 or num_r > 0:
            self.real_seq_length = [l - num_l - num_r for l in self.real_seq_length]
            if self.token_ids is not None:
                self.token_ids = self.token_ids[:, num_l:-num_r]
            for layer_idx in range(len(self.key_cache)):
                self.key_cache[layer_idx] = self.key_cache[layer_idx][..., num_l:-num_r, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][..., num_l:-num_r, :]

    def unbind(self) -> List["SequenceCache"]:

        seq_num = self.get_seq_num()
        keys = []
        values = []
        # iterate over each transformers layer
        for key, value in zip(self.key_cache, self.value_cache):
            keys.append(key.unbind(0))
            values.append(value.unbind(0))

        # [Layer, Seq] -> [Seq, Layer]
        keys = zip(*keys)
        values = zip(*values)

        caches: List[SequenceCache] = []
        for k, v, s in zip(keys, values, self.real_seq_length):
            cache = SequenceCache()
            cache.key_cache = [t.unsqueeze(0) for t in k]
            cache.value_cache = [t.unsqueeze(0) for t in v]
            cache.real_seq_length = [s]
            caches.append(cache)
        assert len(caches) == seq_num

        if self.token_ids is not None:
            token_ids = self.token_ids.unbind(0)
            assert len(token_ids) == seq_num
            for tids, cache in zip(token_ids, caches):
                cache.token_ids = tids.unsqueeze(0)

        if len(self.next_logits) > 0:
            assert len(self.next_logits) == seq_num
            for n, cache in zip(self.next_logits, caches):
                cache.next_logits = [n]

        if len(self.last_texts) > 0:
            assert len(self.last_texts) == seq_num
            for t, cache in zip(self.last_texts, caches):
                cache.last_texts = [t]

        return caches

    def unbind_and_trim(
        self,
        prefix_lengths: List[int],
        max_prefix_len: int,
        input_lengths: List[int],
        max_input_len: int,
    ) -> List["SequenceCache"]:

        caches = self.unbind()

        assert len(prefix_lengths) == len(input_lengths) == len(caches)
        for p, i, cache in zip(prefix_lengths, input_lengths, caches):
            cache.trim(max_prefix_len - p, max_input_len - i)

        return caches

    def expand_seq(self, repeat_times: int) -> "SequenceCache":
        assert self.get_seq_num() == 1, "SequenceCache can only repeat sequence when it contains only one sequence"

        cache = SequenceCache()
        cache.next_logits = self.next_logits * repeat_times
        cache.last_texts = [self.last_texts[0]] * repeat_times  # repeat a list of strings
        cache.real_seq_length = self.real_seq_length * repeat_times
        if self.token_ids is not None:
            cache.token_ids = self.token_ids.expand(repeat_times, -1)
        for key, value in zip(self.key_cache, self.value_cache):
            cache.key_cache.append(key.expand(repeat_times, -1, -1, -1))
            cache.value_cache.append(value.expand(repeat_times, -1, -1, -1))
        return cache

    @classmethod
    def pad_and_stack(cls, seq_caches: Sequence["SequenceCache"], pad_token_id: int = 0) -> "SequenceCache":
        # filter out empty cache
        seq_caches = [sc for sc in seq_caches if sc.key_cache[0].shape[0] > 0]
        if len(seq_caches) == 1:
            return seq_caches[0]
        elif len(seq_caches) <= 0:
            raise ValueError("No cache to pad and stack")

        cache = SequenceCache()
        for sc in seq_caches:
            cache.next_logits.extend(sc.next_logits)
            cache.last_texts.extend(sc.last_texts)
            cache.real_seq_length.extend(sc.real_seq_length)

        max_seq_len = max(cache.real_seq_length)
        max_layer_idx = len(seq_caches[0].key_cache)

        if all(sc.token_ids is not None for sc in seq_caches):
            token_shape = (len(cache.real_seq_length), max_seq_len)
            cache.token_ids = torch.full(token_shape, pad_token_id, dtype=torch.long)
            last_idx = 0
            for sc in seq_caches:
                seq_num, seq_len = sc.token_ids.shape
                if seq_len > 0:
                    cache.token_ids[last_idx:last_idx + seq_num, -seq_len:] = sc.token_ids
                last_idx += seq_num
            assert last_idx == token_shape[0]

        kv_tensors = []
        for layer_idx in range(max_layer_idx):
            # each layer may on different devices
            kv_shape = seq_caches[0].key_cache[layer_idx].shape
            kv_shape = (sum(sc.key_cache[0].shape[0] for sc in seq_caches), kv_shape[1], max_seq_len, kv_shape[-1])
            _to = {
                "device": seq_caches[0].key_cache[layer_idx].device,
                "dtype": seq_caches[0].key_cache[layer_idx].dtype
            }
            key = torch.zeros(kv_shape, **_to)
            value = torch.zeros(kv_shape, **_to)
            last_idx = 0
            for sc in seq_caches:
                seq_num, _, seq_len, _ = sc.key_cache[layer_idx].shape
                if seq_len > 0:
                    key[last_idx:last_idx + seq_num, :, -seq_len:, :] = sc.key_cache[layer_idx]
                    value[last_idx:last_idx + seq_num, :, -seq_len:, :] = sc.value_cache[layer_idx]
                last_idx += seq_num
            assert last_idx == kv_shape[0], f"{last_idx} != {kv_shape[0]}"
            kv_tensors.append((key, value))

        cache.key_cache, cache.value_cache = map(list, zip(*kv_tensors))
        assert len(cache.key_cache) == len(cache.value_cache)
        assert cache.key_cache[0].shape[1] == cache.value_cache[0].shape[1]

        return cache

    def to_legacy_cache(self) -> Optional[DynamicCache]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        if len(self.key_cache) == 0 or any(s == 0 for s in self.key_cache[0].shape):
            return None

        legacy_cache = DynamicCache()
        legacy_cache.key_cache = self.key_cache
        legacy_cache.value_cache = self.value_cache
        return legacy_cache

    def __repr__(self) -> str:
        reprs = []
        reprs.append(f"real_seq_length={self.real_seq_length}")
        if self.token_ids is not None:
            reprs.append(f"token_ids=[{', '.join(str(t.shape) for t in self.token_ids)}]")
        if self.last_texts:
            reprs.append(f"last_texts={self.last_texts}")
        if self.next_logits:
            endings = '] * ' + str(len(self.next_logits)) if self.next_logits else ']'
            reprs.append(f"next_logits=[{repr(self.next_logits[0].shape)}{endings}")
        return f"SequenceCache({', '.join(reprs)})"


class Cacher:
    """A base class that supports caching for a list of sources."""

    def get_cache(self) -> Tuple[Optional[SequenceCache], int]:
        raise NotImplementedError

    def set_cache(self, caches: List[SequenceCache]):  # -> Any:# -> Any:
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


def round_down(n):
    n |= (n >> 1)
    n |= (n >> 2)
    n |= (n >> 4)
    n |= (n >> 8)
    n |= (n >> 16)
    n |= (n >> 32)
    return n - (n >> 1)


class CachePrefixSampler(Sampler[List[int]], Cacher):
    """A sampler that facilitates key-value caching for a list of text segments.

    Consider a batch of data indexed from 0 to 7 with cache level 2. Assume data
    0~3 have the same prefix and 4~7 have the same prefix. We need to yield the
    data 0 and 4 to cache the prefix, and then yield 0~7 to generate with the cache.
    Notes that the data 0 and 4 will be yielded twice in total.

    Args:
        data: The data to sample from.
        total: The total length of data.
        total_prefix_num: The number of prefixes to cache.
        batch_size: The maximum batch size.
        auto_batch_size: Whether to automatically adjust the batch size based on the maximum length of the data.
        index_offset: The  offset of indices to yield.
        """

    def __init__(
        self,
        data: Iterator[Conversation],
        total: int,
        total_prefix_num: int,
        batch_size: int,
        auto_batch_size: bool = False,
        index_offset: int = 0,
    ):

        # split data into (src,) and (src, tgt)
        self.total_prefix_num = total_prefix_num
        self.joined_data = [[] for _ in range(self.total_prefix_num)]
        self.cache_levels = [0] * total
        self.index_offset = index_offset

        # the batch_size for the kvcache is smaller than the batch_size to avoid OOM
        cache_batch_size = (batch_size + 1) // 2
        self.cache_batch_size = [cache_batch_size] * (self.total_prefix_num - 1) + [batch_size]
        self.auto_batch_size = auto_batch_size

        self.cache: Dict[Tuple[int, int], SequenceCache] = dict()
        """The KV-Cache for the prefixes. The key is a tuple of (cache_level, data_idx)."""

        self.next_data_idx = self._get_next_data_idx(data, total)
        self.data_order_with_cache = self._get_data_order(total)

        # initialize for the data iterator
        self.data_idx = 0
        self.last_cache_st = [-1] * self.total_prefix_num
        self.last_cache_ed = [0] * self.total_prefix_num
        self.last_cache: List[Optional[SequenceCache]] = [None] * self.total_prefix_num

    def _get_next_data_idx(self, data, total):
        """`next_data_idx` holds the range of data that has the same prefix at different cache levels. Data in the range from `i` to `next_data_idx[c][i] - 1` has the number of `c` same prefix. If `i` is not in the list of `next_data_idx[c]`, it means `next_data_idx[c][i]` has the default value `i + 1`."""

        next_data_idx = [dict() for _ in range(self.total_prefix_num)]

        last_start_idx = [0 for _ in range(self.total_prefix_num)]
        for s_idx, src in enumerate(data):
            if hasattr(src, "to_model_prompt"):
                src = src.to_model_prompt()
            for p_idx in range(self.total_prefix_num):
                joined_src = "".join(src[:p_idx + 1])
                self.joined_data[p_idx].append(joined_src)
                if s_idx > 0 and joined_src != self.joined_data[p_idx][s_idx - 1]:
                    if last_start_idx[p_idx] + 1 != s_idx:
                        next_data_idx[p_idx][last_start_idx[p_idx]] = s_idx
                    last_start_idx[p_idx] = s_idx
        for p_idx in range(self.total_prefix_num):
            if last_start_idx[p_idx] + 1 != total:
                next_data_idx[p_idx][last_start_idx[p_idx]] = total

        return next_data_idx

    def _get_data_order(self, total):
        """The data indices for each batch."""

        data_order_with_cache: List[List[int]] = []

        # In the case of data indexed from 0 to 7, if batch_size = 4:
        # `data_order_with_cache` will be [[0, 4], [0, 1, 2, 3], [4, 5, 6, 7]]
        # `data_cache_level` will be [0, 1, 1] which means the first batch has
        # cache level 0 and the rest have cache level 1.

        self.data_cache_level = []
        is_cache_ready = [0 for _ in range(self.total_prefix_num)]
        order_idx_by_cache: List[int] = [-1] * self.total_prefix_num

        # hold the average sequence length for the first batch to keep the
        # cuda memory utilization stable
        self.avg_max_len = [None for _ in range(self.total_prefix_num)]

        for data_idx in range(total):
            for i in range(self.total_prefix_num):
                if is_cache_ready[i] <= data_idx:
                    if order_idx_by_cache[i] == -1:
                        data_order_with_cache.append([])
                        self.data_cache_level.append(i)

                        order_idx_by_cache[i] = len(data_order_with_cache) - 1

                    data_order_with_cache[order_idx_by_cache[i]].append(data_idx)

                    if i == self.total_prefix_num - 1:
                        # resolve duplicate data
                        is_cache_ready[i] = data_idx + 1
                    else:
                        is_cache_ready[i] = self.next_data_idx[i].get(data_idx, data_idx + 1)
                    for j in range(i + 1, self.total_prefix_num):
                        if order_idx_by_cache[j] != -1 and order_idx_by_cache[j] < order_idx_by_cache[i]:
                            order_idx_by_cache[j] = -1
                    if self.check_new_batch(data_order_with_cache[order_idx_by_cache[i]], i, data_idx + 1):
                        order_idx_by_cache[i] = -1

        for o in data_order_with_cache:
            o = [i + self.index_offset for i in o]

        return data_order_with_cache

    def check_new_batch(self, queries: List[int], cache_level: int, next_data: int) -> bool:
        """Check the condition to start a new batch."""

        current_batch = len(queries)
        if not self.auto_batch_size:
            return current_batch > self.cache_batch_size[cache_level]
        max_len = max(len(self.joined_data[cache_level][q]) for q in queries)
        if next_data < len(self.joined_data[cache_level]):
            max_len = max(len(self.joined_data[cache_level][next_data]), max_len)

        if self.avg_max_len[cache_level] is None:
            self.avg_max_len[cache_level] = max_len

        available_space = self.cache_batch_size[cache_level] * self.avg_max_len[cache_level]

        batch_size = available_space // max_len
        batch_size = round_down(batch_size)
        return current_batch >= batch_size

    def __len__(self):
        return len(self.data_cache_level)

    def get_cache(self) -> Tuple[Optional[SequenceCache], int]:
        """Get cache for a list of sources. Return None if any source is not cached.

        Return:
            cache (`SequenceCache`): The (left padded) cache for the sources.
            cache_level (`int`): The number of prefixes that are matched in the cache.
        """

        cache_level = self.data_cache_level[self.data_idx]
        if cache_level == 0:
            return None, 0  # do not have prefix

        def lower_bound(i, l) -> Tuple[int, int]:
            max_k = 0
            for k, _ in self.next_data_idx[l].items():
                if k < i:
                    max_k = max(max_k, k)
            return max_k, self.next_data_idx[l].get(max_k, max_k + 1)

        # `last_cache_count` is local variable while `self.last_cache_st`
        # and `self.last_cache_ed` is global variable because one cache
        # might be used by different batches, and we count the repeated
        # time separately for different batches.
        caches = []
        last_cache_count = 0

        for i in self.data_order_with_cache[self.data_idx]:
            #
            #        ↙ latest cache
            #   [Prefix1] [Data1]  ← last_cache_st
            #             [Data2]
            #   [Prefix2] [Data3]  ← current data i == last_cache_ed
            #             [Data4]
            #
            # if the current data (i) is not in the latest cache, we update the
            # latest cache in following steps:
            #  1. save the cache of the last data (i - 1) for return
            #  2. delete the cache of the last data
            #  3. retrieve the cache of the current data
            #  4. update `last_cache_count` and `last_cache`
            if self.last_cache_ed[cache_level] <= i:

                # 1. save the cache of the last data (i - 1) for return
                if last_cache_count > 0:
                    caches.append(self.last_cache[cache_level].expand_seq(last_cache_count))

                # 2. delete the cache of the last data if it exists
                if self.last_cache_st[cache_level] > -1:
                    del self.cache[(cache_level - 1, self.last_cache_st[cache_level])]

                # 3. retrieve the cache of the current data
                #   case 1: [..., st = i, ..., ed]  (if `i` is in `next_data_idx`)
                #   case 2: [..., st = i, ed = i + 1]  (when cache_level == total_prefix_num)
                #   case 3: [..., st, ..., i, ..., ed]  (find with lower_bound)
                self.last_cache_st[cache_level] = i
                self.last_cache_ed[cache_level] = self.next_data_idx[cache_level - 1].get(i, None)
                if self.last_cache_ed[cache_level] is None:
                    if cache_level == self.total_prefix_num:
                        self.last_cache_ed[cache_level] = i + 1
                    else:
                        st, ed = lower_bound(i, cache_level - 1)
                        if i == ed:
                            self.last_cache_ed[cache_level] = i + 1
                        else:
                            self.last_cache_st[cache_level] = st
                            self.last_cache_ed[cache_level] = ed

                # 4. update `last_cache_count` and `last_cache`
                last_cache_count = 1
                if (cache_level - 1, self.last_cache_st[cache_level]) not in self.cache:
                    print(f"Cache not found: {cache_level - 1}, {self.last_cache_st[cache_level]}")
                    print(i, self.last_cache_st[cache_level], self.last_cache_ed[cache_level])
                    print(self.cache.keys())
                    print(self.joined_data[self.total_prefix_num - 1][i - 1])
                    print("====")
                    print(self.joined_data[self.total_prefix_num - 1][i])
                    print("====")
                    print(self.joined_data[self.total_prefix_num - 1][i + 1])

                    print(self.data_cache_level[self.data_idx - 10:self.data_idx - 1])
                    print(self.data_order_with_cache[self.data_idx - 10:self.data_idx - 1])

                    print(self.data_cache_level[self.data_idx])
                    print(self.data_order_with_cache[self.data_idx])

                    print(self.data_cache_level[self.data_idx + 1:self.data_idx + 10])
                    print(self.data_order_with_cache[self.data_idx + 1:self.data_idx + 10])
                    raise ValueError
                self.last_cache[cache_level] = self.cache[(cache_level - 1, self.last_cache_st[cache_level])]
            else:
                last_cache_count += 1
        caches.append(self.last_cache[cache_level].expand_seq(last_cache_count))
        return SequenceCache.pad_and_stack(caches), cache_level

    def set_cache(self, caches: List[SequenceCache]):

        cache_level = self.data_cache_level[self.data_idx]
        for i, cache in zip(self.data_order_with_cache[self.data_idx], caches):
            self.cache[(cache_level, i)] = cache

    def step(self):
        self.data_idx += 1

    def __iter__(self) -> Iterator[List[int]]:
        self.data_idx = 0
        yield from self.data_order_with_cache

    def __repr__(self) -> str:
        return f"CachePrefixSampler(cache_batch_size={self.cache_batch_size}, total_prefix_num={self.total_prefix_num})"
