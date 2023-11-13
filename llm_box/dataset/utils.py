from logging import getLogger
from os.path import normpath
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ..model.model import Model
from ..utils import DatasetArguments, import_subclass
from .dataset import Dataset
from .raw_dataset_loader import RAW_DATASET_COLLECTIONS, load_raw_dataset

logger = getLogger(__name__)


def resolve_dataset_details(
    dataset_name_or_path: str,
    subsets: Optional[Union[str, List[str]]],
) -> Tuple[Optional[str], str, Optional[List[str]]]:
    """
    Resolve dataset details based on the provided path and subsets.

    Args:
        dataset_name_or_path (str): Path or name of the dataset.
        subsets (Optional[Union[str, List[str]]]): Subsets to be loaded.

    Returns:
        tuple: A tuple containing dataset_name, dataset_path, and subsets.
    """
    if isinstance(subsets, (str, list)):
        # automatically determine dataset_name
        return (
            None,
            dataset_name_or_path,
            [subsets] if isinstance(subsets, str) else subsets,
        )

    # load from config, e.g. `copa` and `race`
    for _collection, (_indexable, _subsets) in RAW_DATASET_COLLECTIONS.items():
        if dataset_name_or_path in _subsets and _indexable:
            # index subset name directly, e.g. `copa`
            return dataset_name_or_path, _collection, [dataset_name_or_path]

        elif _collection == dataset_name_or_path:
            # to avoid duplicate subsets like `"all"` in "race" dataset
            return dataset_name_or_path, _collection, _subsets

    # automatically determine dataset_name and subsets
    return None, dataset_name_or_path, None


def load_dataset(
    args: DatasetArguments,
    model: Model,
    dataset_name_or_path: Union[str, Path],
    subsets: Optional[Union[str, List[str]]] = None,
    split: Optional[str] = None,
    methods: Optional[Union[str, List[str]]] = None,
) -> List[Dataset]:
    r"""Load the corresponding dataset classes of a single dataset or the subsets of a dataset collection. Raw dataset will be automatically loaded and formatted into `Dataset` class.

    Args:
        - `args (DatasetArguments)`: Configuration parameters required for dataset processing.

        - `model (Model)`: The model for which the dataset is being loaded. This parameter can be used to tailor the dataset processing to the specific requirements or characteristics of the model.

        - `dataset_name_or_path (Union[str, Path])`: The name of the dataset or the file path to the dataset. If a name is provided, the dataset is expected to be a known dataset in huggingface datasets. If a path is provided, it is expected to be a valid file path for huggingface `load_from_disk` method.

        - `subsets (Optional[Union[str, List[str]]], optional)`: Specific subsets of the dataset to load. This can be a single subset name or a list of subset names. If None, all available subsets of the dataset are loaded. Defaults to None.

        - `split (Optional[str], optional)`: The specific split of the dataset to load (e.g., 'train', 'test', 'validation'). If None, all splits are loaded. Defaults to None.

        - `methods (Optional[Union[str, List[str]]], optional)`: The methods to use for loading the dataset. This can include methods like 'load_datasets', 'load_from_disk', etc. If None, default methods are used based on the dataset and its format. Defaults to None.

    Returns:
        List[Dataset]: A list of our class for dataset.
    """
    # find the relative path from `main`
    dataset_name_or_path = str(normpath(dataset_name_or_path))
    dataset_name, dataset_path, subsets = resolve_dataset_details(dataset_name_or_path, subsets)

    # load all subsets from dataset
    logger.debug(f"Loading raw dataset from {dataset_path} - {subsets}")
    is_subset = True
    for n in [dataset_name, dataset_path.split("/")[-1]]:
        if n is None:
            continue
        try:
            dataset_cls = import_subclass(
                module_path='llm_box.dataset.' + n,
                metaclass_type=Dataset,
            )
            logger.debug(f"Dataset class `{dataset_cls.__name__}` loaded.")
            is_subset = False
            break
        except Exception:
            continue

    # load raw datasets
    if not is_subset and methods is None:
        methods = dataset_cls.load_methods
    elif methods is None:
        methods = ['load_datasets', 'load_from_disk']
    raw_datasets = load_raw_dataset(dataset_path=dataset_path, subsets=subsets, split=split, methods=methods)

    # wrap raw dataset into LLMDataset
    llm_datasets = []
    for subset, raw_dataset in raw_datasets.items():
        if is_subset:
            # we need to import the dataset class for each subset
            dataset_cls = import_subclass(
                module_path='llm_box.dataset.' + subset,
                metaclass_type=Dataset,
            )
        dataset = dataset_cls(
            args=args,
            model=model,
            subset=subset,
            raw_dataset=raw_dataset,
        )
        llm_datasets.append(dataset)

    return llm_datasets
