import importlib
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import getLogger
from typing import TYPE_CHECKING, Dict, Iterator, List, Set, Type

import openai
from datasets import DownloadConfig, get_dataset_config_names

from ..metric import GPTEval
from .dataset import Dataset, DatasetCollection
from .enum import DATASET_ALIASES
from .utils import accepts_subset

if TYPE_CHECKING:
    # solve the circular import
    from ..model.model import Model
    from ..utils import DatasetArguments

logger = getLogger(__name__)

ABSTRACT_DATASET = {"Dataset", "GenerationDataset", "MultipleChoiceDataset"}


def _import_dataset_class(dataset_name: str) -> Type[Dataset]:

    module_path = __package__ + "." + dataset_name
    module = importlib.import_module(module_path)
    clsmembers = inspect.getmembers(module, inspect.isclass)

    for name, obj in clsmembers:
        if issubclass(obj, Dataset) and name not in ABSTRACT_DATASET:
            logger.debug(f"Dataset class `{name}` imported from `{module_path}`.")
            return obj

    raise ValueError(
        f"Cannot find dataset class with name {dataset_name} in module {module_path}. "
        "Make sure the dataset class defines `name` attribute properly."
    )


def import_dataset_classes(dataset_name: str) -> List[Type[Dataset]]:
    if dataset_name in DATASET_ALIASES:
        logger.info("Loading dataset aliases: %s -> %s", dataset_name, DATASET_ALIASES[dataset_name])
        return [_import_dataset_class(alias) for alias in DATASET_ALIASES[dataset_name]]
    else:
        return [_import_dataset_class(dataset_name)]


def get_subsets(dataset_name: str, dataset_classes: List[Type[Dataset]], offline: bool = False) -> List[Set[str]]:

    available_subsets = set()
    available_subsets_by_cls = []

    if not offline:
        for dataset_cls in dataset_classes:

            # TODO: fix Tuple[()]
            hf_path = dataset_cls.load_args[0] if len(dataset_cls.load_args) > 0 else dataset_name
            download_config = DownloadConfig(use_etag=False)
            try:
                s = get_dataset_config_names(hf_path, download_config=download_config, trust_remote_code=True)
            except Exception as e:
                logger.info(f"Failed when trying to get_dataset_config_names: {e}")
                s = []

            if s == ["default"]:
                s = []

            available_subsets.update(s)
            available_subsets_by_cls.append(set(s))

    # for wmt, en-xx and xx-en are both supported
    if "wmt" == dataset_name:
        for subset in available_subsets.copy():
            if subset.endswith("-en"):
                available_subsets.add("en-" + subset.split("-")[0])

    # GPTEval requires openai-gpt
    if any(isinstance(m, GPTEval) for m in dataset_cls.metrics) and openai.api_key is None:
        raise ValueError(
            "OpenAI API key is required for GPTEval metrics. Please set it by passing a `--openai_api_key` or through environment variable `OPENAI_API_KEY`."
        )

    # load dataset
    if "squad_v2" in dataset_name:
        dataset_cls.load_args = ("squad_v2",)

    for idx, a in enumerate(available_subsets_by_cls):
        available_subsets_by_cls[idx] = a.intersection(available_subsets)
        available_subsets -= a

    return available_subsets_by_cls


def get_cmd_subset_names(cmd_subset_names: Set[str], dataset_cls: "Dataset") -> Set[str]:
    """Get the subset names from the command line arguments. If the subset name is a category, expand it to the actual subset names."""
    results = set()
    for cmd_subset in cmd_subset_names:
        if cmd_subset.startswith("[") and cmd_subset.endswith("]"):
            if dataset_cls.categorized_subsets is None:
                continue
            categorized_subsets = dataset_cls.categorized_subsets[cmd_subset[1:-1]]
            logger.info(f"Expanding category `{cmd_subset}` to `{categorized_subsets}`")
            results.update(categorized_subsets)
        else:
            results.add(cmd_subset)
    return results


def load_dataset(dataset_name: str,
                 args: "DatasetArguments",
                 model: "Model",
                 threading: bool = True) -> Iterator[Dict[str, Dataset]]:
    r"""Load corresponding dataset class.

    Args:
        args (Namespace): The global configurations.
        model (Model): Our class for model.

    Returns:
        Dataset: Our class for dataset.
    """

    dataset_classes = import_dataset_classes(dataset_name)
    available_subsets_by_cls = get_subsets(dataset_name, dataset_classes, offline=args.dataset_path is not None)

    for dataset_cls, available_subsets in zip(dataset_classes, available_subsets_by_cls):

        cmd_subset_names = get_cmd_subset_names(args.subset_names, dataset_cls)
        if len(args.subset_names) > 0 and len(cmd_subset_names) == 0:
            continue

        # for mmlu and race dataset, remove "all" subset by default
        if dataset_name in {"mmlu", "race"} and len(cmd_subset_names) == 0:
            available_subsets.remove("all")

        # if dataset not in huggingface, allow to manually specify subset_names
        if len(available_subsets) and not available_subsets.issuperset(cmd_subset_names):
            na = cmd_subset_names - available_subsets
            raise ValueError(
                f"Specified subset names {na} are not available. Available ones of {dataset_name} are: {available_subsets}"
            )

        # use specified subset_names if available
        subset_names = cmd_subset_names or available_subsets
        logger.debug(
            f"{dataset_name} - available_subsets: {available_subsets}, load_args: {dataset_cls.load_args}, final subset_names: {subset_names}"
        )

        if len(subset_names) > 1 and accepts_subset(dataset_cls.load_args, overwrite_subset=len(cmd_subset_names) > 0):
            # race:middle,high (several subsets) , super_glue (all the subsets)
            logger.debug(f"Loading subsets of dataset `{dataset_name}`: " + ", ".join(subset_names))
            if threading and len(subset_names) >= 2:
                first_dataset = subset_names.pop()
                first_dataset = (
                    dataset_name + ":" + first_dataset, dataset_cls(dataset_name, args, model, first_dataset)
                )
                logger.info(f"Loading other {len(subset_names)} subsets ...")
                logging.disable(logging.INFO)
                with ThreadPoolExecutor(max_workers=len(subset_names)) as executor:
                    res = [
                        executor.submit(
                            lambda s: (dataset_name + ":" + s, dataset_cls(dataset_name, args, model, s)), s
                        ) for s in subset_names
                    ]
                datasets = [first_dataset] + [f.result() for f in as_completed(res)]
                datasets = dict(sorted(datasets, key=lambda x: x[0]))
                logging.disable(logging.NOTSET)
            else:
                datasets = {
                    dataset_name + ":" + s: dataset_cls(dataset_name, args, model, s)
                    for s in sorted(subset_names)
                }
            yield datasets

        elif len(subset_names) == 1 and len(available_subsets) != 1 and accepts_subset(
            dataset_cls.load_args, overwrite_subset=len(cmd_subset_names) > 0, subset=next(iter(subset_names))
        ):
            # in some cases of get_dataset_config_names() have only one subset, loading dataset with the a subset name is not allowed in huggingface datasets library
            # len(available_subsets) == 0 means a special case, like wmt
            # race:middle (one of the subsets), coqa (default)
            subset_name = next(iter(subset_names))
            logger.debug(f"Loading subset of dataset `{dataset_name}:{subset_name}`")
            yield {dataset_name + ":" + subset_name: dataset_cls(dataset_name, args, model, subset_name)}

        else:
            # copa (super_glue:copa) or anli
            logger.debug(f"Loading dataset `{dataset_name}`")
            yield {dataset_name: dataset_cls(dataset_name, args, model)}


def load_datasets(args: "DatasetArguments", model: "Model", threading: bool = True) -> DatasetCollection:
    datasets = []
    for d in args.dataset_names:
        datasets.extend(load_dataset(d, args, model, threading))
    datasets = {k: v for d in datasets for k, v in d.items()}
    dataset_collection = DatasetCollection(datasets)
    logger.info(f"Evaluation datasets: {dataset_collection}")
    return dataset_collection
