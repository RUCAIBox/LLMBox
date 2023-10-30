import os
import traceback
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any
import inspect

import datasets
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

logger = getLogger(__name__)

raw_dataset_config = {
    "super_glue": {"axb", "axg", "boolq", "cb", "copa", "multirc", "record", "wic", "wsc"},
}

_raw_dataset_loader_registry = dict()

_required_parameters = {'dataset_path', 'subset_names', 'split'}


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


def get_dataset_subset_names(dataset_path, by_split="test", local=False, **kwargs) -> List[str]:
    """Get the list of available subset names (called config names in huggingface's library) for a particular dataset such as MMLU."""

    if not local:
        try:
            return datasets.get_dataset_config_names(dataset_path, **kwargs)
        except Exception as e:
            logger.debug(
                f"Failed to fetch dataset config names from huggingface "
                f"with `{dataset_path}`: {e}. Trying with local files."
            )

    files = os.listdir(os.path.join(dataset_path, by_split))
    filter = lambda f: f.endswith(f"_{by_split}.csv") and not f.startswith(".")
    dataset_subset = {f.split("_test.csv")[0] for f in files if filter(f)}

    dataset_subset = sorted(dataset_subset)
    return dataset_subset


def load_dataset(
    dataset_path: Union[str, Path] = None,
    subset_names: Optional[Union[str, List[str]]] = None,
    split: Optional[str] = None,
    *,
    load_dataset_kwargs: Optional[Dict] = None,
    **ignored_kwargs
) -> Dict[str, Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]]:
    if load_dataset_kwargs is None:
        load_dataset_kwargs = dict()
    logger.debug(f"load_dataset_kwargs: {load_dataset_kwargs}")

    if isinstance(subset_names, str):
        return {
            "default": datasets.load_dataset(path=dataset_path, name=subset_names, split=split, **load_dataset_kwargs)
        }
    else:
        if subset_names is None:
            subset_names = datasets.get_dataset_config_names(dataset_path)
        return {
            subset: datasets.load_dataset(path=dataset_path, name=subset, split=split, **load_dataset_kwargs)
            for subset in subset_names
        }


def load_from_disk(
    dataset_path: Union[str, Path] = None,
    subset_names: Optional[Union[str, List[str]]] = None,
    split: Optional[str] = None,
    *,
    load_from_disk_kwargs: Optional[Dict] = None,
    **ignored_kwargs
) -> Dict[str, Union[DatasetDict, Dataset]]:
    if load_from_disk_kwargs is None:
        load_from_disk_kwargs = dict()

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
    disable_download: bool = False,
    *,
    methods: Union[str, List[str]] = "load_dataset",
    load_dataset_kwargs: Optional[Dict] = None,
    load_from_disk_kwargs: Optional[Dict] = None,
    user_loadder_kwargs: Optional[Dict] = None
) -> Dict[str, Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]]:
    """`load_raw_dataset` provides a unified way to load raw datasets from huggingface or locally. For datasets without subsets, the entire dataset will be used as `default`.

    Example:

    Load CoPA subset of SuperGLUE dataset using `datasets.load_dataset`:
    >>> load_raw_dataset(super_glue", "copa")
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
    if load_dataset_kwargs is None:
        load_dataset_kwargs = dict()
    if load_from_disk_kwargs is None:
        load_from_disk_kwargs = dict()
    if user_loadder_kwargs is None:
        user_loadder_kwargs = dict()

    # logging
    msg = f"Loading dataset {dataset_path}" + \
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

    logger.debug(f"disable_download: {disable_download}")
    if disable_download:
        load_dataset_kwargs['download_config'] = datasets.DownloadConfig(
            local_files_only=True, storage_options={'hf': {
                'token': None,
                'endpoint': ''
            }}
        )

    # try load with methods
    raw_dataset = None
    for loadder_fn in methods:
        try:
            raw_dataset = loadder_fn(
                dataset_path=dataset_path,
                subset_names=subset_names,
                split=split,
                load_dataset_kwargs=load_dataset_kwargs,
                load_from_disk_kwarg=load_from_disk_kwargs,
                user_loadder_kwargs=user_loadder_kwargs,
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
