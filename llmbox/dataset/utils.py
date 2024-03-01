import json
import os
import re
from importlib.machinery import SourceFileLoader
from logging import getLogger
from os.path import abspath
from typing import Callable, List, Optional, Tuple, Union

import datasets

logger = getLogger(__name__)


def accepts_subset(
    load_args: Union[Tuple[str], Tuple[str, str], Tuple[()]],
    overwrite_subset: bool = True,
    subset: str = "",
    disable_warning: bool = False,
) -> bool:
    if len(load_args) == 2 and isinstance(load_args[1], str):
        if overwrite_subset:
            if not disable_warning:
                logger.warning(
                    f"Dataset class already has a subset '{load_args[1]}' to load. Overwriting it with '{subset}'.",
                    stacklevel=2,
                )
        else:
            return load_args[1] == subset
    # len(load_args) == 1 means accept subset and len(load_args) == 0 means special case like wmt
    return True


def get_raw_dataset_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    subset_name: Optional[str],
    load_args: Optional[Union[Tuple[str], Tuple[str, str], Tuple[()]]],
    return_msg: bool = False,
    use_etag: bool = False,
) -> Union[
    Callable[[Optional[str]], datasets.Dataset],
    Tuple[Callable[[str], datasets.Dataset], str],
]:
    """Get the function to load the raw dataset from huggingface (if `load_args` is not None) or local path (if `dataset_path` is not None).

    ```python
    load_fn = get_raw_dataset_loader(...)
    evaluation_data = load_fn(split="test")
    example_data = load_fn(split="train")
    ```

    Search path:
    - huggingface `load_dataset(*load_args)`
    - huggingface `load_dataset(load_args[0], subset_name)`
    - local repo or directory `"{dataset_path}"`
    - local repo or directory `"{dataset_path}/{subset_name}"`
    - local repo or directory `"{dataset_path}/{dataset_name}"`
    - local file pattern `"{dataset_path}".format(subset=subset_name, split=split)`

    """
    if subset_name:
        dataset_msg = f"{dataset_name}:{subset_name}"
    else:
        dataset_msg = f"{dataset_name}"
    msg = f"Loading raw dataset `{dataset_msg}`"
    load_fn = None

    download_config = datasets.DownloadConfig(use_etag=use_etag)

    # if `dataset_path` is not None, load from local path
    if dataset_path is not None:
        dataset_path = abspath(dataset_path)
        msg += f" from local path `{dataset_path}`"
        if subset_name is None and len(load_args) > 1 and load_args[1] is not None:
            subset_name = load_args[1]

        # load from a cloned repository from huggingface
        if os.path.exists(os.path.join(dataset_path, "dataset_infos.json")):
            infos = json.load(open(os.path.join(dataset_path, "dataset_infos.json")))

            # find the correct subset
            if dataset_name in infos:

                logger.debug(f"Loading from a cloned repository: {dataset_path}, {dataset_name}")
                def load_fn(split):
                    return datasets.load_dataset(
                        dataset_path,
                        dataset_name,
                        split=split,
                        trust_remote_code=True,
                        download_config=download_config
                    )

            elif subset_name in infos:

                logger.debug(f"Loading from a cloned repository: {dataset_path}, {subset_name}")
                def load_fn(split):
                    return datasets.load_dataset(
                        dataset_path,
                        subset_name,
                        split=split,
                        trust_remote_code=True,
                        download_config=download_config,
                    )

            else:
                raise ValueError(
                    f"Cannot find `{subset_name}` subset of `{dataset_name}` dataset in `{dataset_path}`. Available subsets: {infos.keys()}"
                )

        # load from a local directory
        elif os.path.exists(os.path.join(dataset_path, "dataset_dict.json")):

            logger.debug(f"Loading from a local directory: {dataset_path}")
            def load_fn(split):
                return datasets.load_from_disk(dataset_path)[split]

        # load from a local directory with subset
        elif subset_name is not None and os.path.exists(os.path.join(dataset_path, subset_name, "dataset_dict.json")):

            new_dataset_path = os.path.join(dataset_path, subset_name)
            logger.debug(f"Loading from a local directory with subset: {new_dataset_path}")
            def load_fn(split):
                return datasets.load_from_disk(new_dataset_path)[split]

        # for those datasets that is in huggingface but should be downloaded manually
        elif os.path.isdir(dataset_path):

            logger.debug(f"Loading from a manually-downloaded dataset: {dataset_name}, {subset_name}")
            def load_fn(split):
                return datasets.load_dataset(
                    dataset_name,
                    subset_name,
                    split=split,
                    data_dir=dataset_path,
                    trust_remote_code=True,
                    download_config=download_config,
                )

        # load from a file
        else:
            subset_name = subset_name or ""
            r_subset = re.compile(r"{subset}")
            r_postfix = re.compile(r"\[.*\].*$")
            r_split = re.compile(r"{split}")

            logger.debug(f"Loading from a file: {dataset_path}")
            def load_fn(split):
                dataset_file_path = r_subset.sub(subset_name, dataset_path)
                if split:
                    split = r_postfix.sub("", split)
                    dataset_file_path = r_split.sub(split, dataset_file_path)

                logger.debug(f"Searching dataset file: {dataset_file_path}")
                if os.path.exists(dataset_file_path):
                    data = load_raw_dataset_from_file(dataset_file_path)
                    if not split or split not in data:
                        return data
                    return data[split]

                raise ValueError(f"Cannot find raw dataset `{dataset_msg}` in `{dataset_path}`.")

    # load from Hugging Face Hub
    elif load_args is not None:
        # specify the dataset name if if its not specified in `dataset.load_args` (e.g. translation)
        if len(load_args) == 0:
            load_args = (dataset_name,)
        # trying to load a subset if its not specified in `dataset.load_args` (e.g. `load_args=("mmlu",)`
        if accepts_subset(load_args, subset=subset_name, disable_warning=True) and subset_name is not None:
            # ignore load_args[1], because if it is specified, it is equivalent to `subset_name`
            load_args = (load_args[0], subset_name)
        elif subset_name is not None:
            raise ValueError(
                f"Failed to specify `{subset_name}` subset since dataset `{dataset_name}` already has defined one to load ({', '.join(load_args)}). Please use `{dataset_name}`."
            )

        # for wmt, en-xx and xx-en refer to the same subset, xx-en
        if "wmt" in dataset_name and subset_name.startswith("en"):
            load_args = (dataset_name, subset_name.split("-")[1] + "-en")

        msg += f" from huggingface ({', '.join(load_args)})"

        def load_fn(split):
            return datasets.load_dataset(
                *load_args, split=split, trust_remote_code=True, download_config=download_config
            )

    if load_fn is None:
        raise ValueError(
            f"Failed to load dataset `{dataset_msg}`. Please check if the dataset exists in huggingface or local path."
        )

    def informative_load_fn(split=None) -> datasets.Dataset:
        try:
            return load_fn(split=split)
        except KeyError as e:
            raise ValueError(f"Cannot find split `{split}` in `{dataset_msg}`.") from e

    if return_msg:
        return informative_load_fn, msg
    return informative_load_fn


def load_raw_dataset_from_file(dataset_file_path: str) -> datasets.Dataset:
    """Load huggingface dataset from file."""

    if dataset_file_path.endswith((".jsonl", ".json")):
        return datasets.Dataset.from_json(dataset_file_path)
    elif dataset_file_path.endswith(".csv"):
        return datasets.Dataset.from_csv(dataset_file_path)
    elif dataset_file_path.endswith(".txt"):
        return datasets.Dataset.from_text(dataset_file_path)
    elif dataset_file_path.endswith(".py"):
        module = SourceFileLoader("source_dataset", dataset_file_path).load_module()
        objects = [getattr(module, obj) for obj in dir(module) if not obj.startswith("_")]
        if len(objects) == 1:

            def generator():
                yield from objects[0]

            return datasets.Dataset.from_generator(generator)

    raise ValueError(
        f"Cannot find raw dataset from file {dataset_file_path}. Supported formats: .jsonl, .json, .csv, .txt, .py"
    )
