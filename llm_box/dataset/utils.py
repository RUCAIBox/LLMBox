import json
import os
from logging import getLogger
from os.path import abspath
from typing import Callable, Union, Optional, Tuple
import re

import datasets as ds

logger = getLogger(__name__)

EXTENDED_SEARCH_PATHS = [
    "/{subset}",
    "/{split}",
    "/{subset}/{split}",
    "/{split}/{subset}",
]


def _get_raw_dataset_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    subset_name: Optional[str],
    load_args: Optional[Union[Tuple[str], Tuple[str, str]]],
) -> Tuple[Callable[[str], ds.Dataset], str]:
    """Get the function to load the raw dataset from huggingface (if `load_args` is not None) or local path (if `dataset_path` is not None).

    ```python
    load_fn = _get_raw_dataset_loader(...)
    evaluation_data = load_fn(split="test")
    example_data = load_fn(split="train")
    ```

    Search path:
    - huggingface `load_dataset(*load_args)`
    - huggingface `load_dataset(*load_args, subset_name)`
    - local repo or directory `"{dataset_path}"`
    - local repo or directory `"{dataset_path}/{subset_name}"`
    - local repo or directory `"{dataset_path}/{dataset_name}"`
    - local file pattern `"{dataset_path}".format(subset=subset_name, split=split)`

    """
    msg = f"Loading raw dataset `{dataset_name}:{subset_name}`"
    load_fn = None
    if dataset_path is not None:
        dataset_path = abspath(dataset_path)
        msg += f" from local path `{dataset_path}`"
        if os.path.exists(dataset_path + "/dataset_infos.json"):
            infos = json.load(open(dataset_path + "/dataset_infos.json"))
            if dataset_name in infos:
                load_fn = lambda split: ds.load_dataset(dataset_path, dataset_name, split=split)  # type: ignore
            elif subset_name in infos:
                load_fn = lambda split: ds.load_dataset(dataset_path, subset_name, split=split)  # type: ignore
            else:
                raise ValueError(
                    f"Cannot find dataset `{dataset_name}:{subset_name}` in `{dataset_path}`. Available subsets: {infos.keys()}"
                )
        elif os.path.exists(dataset_path + "/dataset_dict.json"):
            load_fn = lambda split: ds.load_from_disk(dataset_path)[split]
        elif isinstance(subset_name, str) and os.path.exists(f"{dataset_path}/{subset_name}/dataset_dict.json"):
            load_fn = lambda split: ds.load_from_disk(dataset_path + "/" + subset_name)[split]
        else:
            subset_name = subset_name or ""
            supported_formats = (".jsonl", ".json", ".csv", ".txt")

            def load_fn(split):
                search_paths = [""]
                if not dataset_path.endswith(supported_formats):
                    search_paths += EXTENDED_SEARCH_PATHS
                for search_path in search_paths:
                    dataset_file_path = dataset_path.rstrip("/") + search_path

                    dataset_file_path = re.sub(r"{subset}", subset_name, dataset_file_path)
                    dataset_file_path = re.sub(r"{split}", split, dataset_file_path)

                    logger.debug(f"Searching dataset file: {dataset_file_path}")
                    if os.path.exists(dataset_file_path):
                        data = load_raw_dataset_from_file(dataset_file_path)
                        if split not in data:
                            logger.warning(f"Cannot find split `{split}` in `{dataset_file_path}`.")
                            return data
                        return data[split]

                raise ValueError(f"Cannot find dataset `{dataset_name}:{subset_name}` in `{dataset_path}`.")

    elif load_args is not None:
        if len(load_args) == 1 and isinstance(subset_name, str):
            load_args = load_args + (subset_name,)
        elif isinstance(subset_name, str):
            raise ValueError(
                f"Failed to specify {subset_name} subset since dataset `{dataset_name}` already has defined one to load ({', '.join(load_args)}). Please use `{dataset_name}`."
            )
        msg += f" from huggingface ({', '.join(load_args)})"
        load_fn = lambda split: ds.load_dataset(*load_args, split=split)  # type: ignore

    if load_fn is None:
        raise ValueError(
            f"Failed to load dataset `{dataset_name}:{subset_name}`. Please check if the dataset exists in huggingface or local path."
        )

    return load_fn, msg


def load_raw_dataset_from_file(dataset_file_path: str) -> ds.Dataset:
    """Load huggingface dataset from file."""

    if dataset_file_path.endswith((".jsonl", ".json")):
        return ds.Dataset.from_json(dataset_file_path)  # type: ignore
    elif dataset_file_path.endswith(".csv"):
        return ds.Dataset.from_csv(dataset_file_path)  # type: ignore
    elif dataset_file_path.endswith(".txt"):
        return ds.Dataset.from_text(dataset_file_path)  # type: ignore
    else:
        raise ValueError(
            f"Cannot load dataset from file {dataset_file_path}. Supported formats: .jsonl, .json, .csv, .txt"
        )
