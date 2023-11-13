import inspect
import os
import traceback
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import datasets
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

__all__ = ["RAW_DATASET_COLLECTIONS", "load_raw_dataset", "register_raw_dataset_loader", "list_raw_dataset_loader"]

logger = getLogger(__name__)

RAW_DATASET_COLLECTIONS = {
    "super_glue": {"axb", "axg", "boolq", "cb", "copa", "multirc", "record", "wic", "wsc"},
}

_raw_dataset_loader_registry = dict()

_required_parameters = {'dataset_path', 'subset_names', 'split'}
"""Parameters required for raw dataset loaders."""


def register_raw_dataset_loader(fn: Callable[..., Any]) -> Callable[..., Any]:
    name = fn.__name__

    # parameters check
    fn_sig = inspect.signature(fn)
    missing_parameters = _required_parameters - set(fn_sig.parameters)
    if len(missing_parameters) > 0:
        raise TypeError(f"Raw dataset loader {name} should contain parameters {missing_parameters}.")

    # register
    if name in _raw_dataset_loader_registry:
        raise ValueError('Raw dataset loader {name} has been registered.')
    _raw_dataset_loader_registry[fn.__name__] = fn

    return fn


def list_raw_dataset_loader() -> List[str]:
    return list(_raw_dataset_loader_registry.keys())


def load_dataset(
    dataset_path: Union[str, Path] = None,
    subset_names: Optional[Union[str, List[str]]] = None,
    split: Optional[str] = None,
) -> Dict[str, Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]]:
    """A wrapper for `datasets.load_dataset` to load raw datasets from huggingface. If `subset_names` is not specified, all subsets will be loaded.

    Returns:
        - `Dict[str, Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]`:
        A dictionary of datasets, with subset names as keys. If it fails to infer the subset names, the key will be `'default'`.
    """
    if isinstance(subset_names, str):
        # TODO: determine whether it should return `subset_names` or `"default"`
        return {subset_names: datasets.load_dataset(path=dataset_path, name=subset_names, split=split)}
    else:
        if subset_names is None:
            subset_names = datasets.get_dataset_config_names(dataset_path)
        return {subset: datasets.load_dataset(path=dataset_path, name=subset, split=split) for subset in subset_names}


def load_from_disk(
    dataset_path: Union[str, Path] = None,
    subset_names: Optional[Union[str, List[str]]] = None,
    split: Optional[str] = None,
) -> Dict[str, Union[DatasetDict, Dataset]]:
    """A wrapper for `datasets.load_from_disk` to load raw datasets from local disk. If `subset_names` is not specified, all subsets will be loaded.

    Returns:
        - `Dict[str, Union[DatasetDict, Dataset]]`: A dictionary of datasets, with subset names as keys. If it fails to infer the subset names, the key will be `'default'`.
    """
    if not isinstance(subset_names, list):
        dataset_path = subset_names | dataset_path

    dataset = datasets.load_from_disk(dataset_path=dataset_path)
    if split is not None:
        dataset = dataset[split]
    return {"default": dataset}


register_raw_dataset_loader(load_dataset)
register_raw_dataset_loader(load_from_disk)


def load_raw_dataset(
    dataset_path: Union[str, Path] = None,
    subset_names: Optional[Union[str, List[str]]] = None,
    split: Optional[str] = None,
    *,
    methods: Union[str, List[str]] = "load_dataset",
) -> Dict[str, Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]]:
    """`load_raw_dataset` provides a unified way to load raw datasets from huggingface or locally. For datasets without subsets, the entire dataset will be used as `default`.

    Example:

    Load CoPA subset of SuperGLUE dataset using `datasets.load_dataset`:
    >>> load_raw_dataset("super_glue", "copa")
    {"copa": DatasetDict(...)}

    Or simply specify `"copa"` (supports SuperGLUE):
    >>> load_raw_dataset("copa")
    {"copa": DatasetDict(...)}

    Load all subsets of MMLU dataset using user registered dataset loader:
    >>> load_raw_dataset("path/to/mmlu", methods="load_origin_mmlu")
    {"abstract_algebra": DatasetDict(...), ...}

    Load CoPA subset of SuperGLUE dataset using `datasets.load_from_disk`:
    >>> load_raw_dataset("path/to/copa", methods="load_from_disk")
    {"default": DatasetDict(...)}

    Load a port of CoPA from `gimmaru/super_glue-copa`:
    >>> load_raw_dataset("gimmaru/super_glue-copa")
    {"default": DatasetDict(...)}
    """
    # normalize parameters
    if isinstance(subset_names, str):
        subset_names = [subset_names]
    if isinstance(methods, str):
        methods = [methods]
    dataset_path = str(os.path.normpath(dataset_path))
    assert "," not in dataset_path and ":" not in dataset_path, "Dataset path should not contain comma or colon."
    assert subset_names is None or all(["," not in n and ":" not in n
                                        for n in subset_names]), "Subset names should not contain comma or colon."

    # logging
    msg = f"Loading raw dataset {dataset_path}" + \
        ("" if subset_names is None else ":" + ",".join(subset_names)) + \
        (" with methods " + ", ".join(methods))
    logger.info(msg)

    # map method names into functions
    not_registered = set(methods) - set(_raw_dataset_loader_registry.keys())
    if len(not_registered) > 0:
        registerd = "{" + ", ".join(_raw_dataset_loader_registry.keys()) + "}"
        raise ValueError(
            f"Raw dataset loading methods {not_registered} not registered in {registerd}. Consider register with `register_raw_dataset_loader`."
        )
    methods = [_raw_dataset_loader_registry[m] for m in methods]

    # try load with methods
    raw_dataset = None
    for loadder_fn in methods:
        try:
            raw_dataset = loadder_fn(
                dataset_path=dataset_path,
                subset_names=subset_names,
                split=split,
            )
        except Exception as e:
            logger.info(f"{loadder_fn.__name__}: {e}")
            logger.debug("\n" + traceback.format_exc())
            continue

    if raw_dataset is None:
        raise RuntimeError(
            "Failed to load from huggingface or local disk. "
            "Please check the dataset path or implement your own dataset loader."
        )
    return raw_dataset
