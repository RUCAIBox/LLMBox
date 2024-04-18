import importlib
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import getLogger
from typing import TYPE_CHECKING, Dict, Iterator, List, Set, Type

import openai
from datasets import DownloadConfig, get_dataset_config_names
from tqdm import tqdm

from utilization.metric.pass_at_k import PassAtK

from ..metric import GPTEval
from ..utils.catch_error import catch_error
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


def get_subsets(
    dataset_name: str,
    dataset_classes: List[Type[Dataset]],
    args: "DatasetArguments",
    offline: bool = False
) -> List[Set[str]]:

    available_subsets = set()
    available_subsets_by_cls = []

    if not offline:
        for dataset_cls in dataset_classes:

            # dynamically set load_args for squad and wmt datasets, in order to support squad_v2 and wmt series datasets
            if not dataset_cls.load_args:
                dataset_cls.load_args = (dataset_name,)

            download_config = DownloadConfig(use_etag=False)
            try:
                s = get_dataset_config_names(
                    dataset_cls.load_args[0], download_config=download_config, trust_remote_code=True
                )
            except Exception as e:
                logger.info(f"Failed when trying to get_dataset_config_names: {e}")
                s = []

            if s == ["default"]:
                s = []

            for m in getattr(dataset_cls, "metrics", []):
                if isinstance(m, GPTEval) and openai.api_key is None:
                    raise ValueError(
                        "OpenAI API key is required for GPTEval metrics. Please set it by passing a `--openai_api_key` or through environment variable `OPENAI_API_KEY`."
                    )
                if isinstance(m, PassAtK) and args.pass_at_k is None:
                    raise ValueError(
                        "PassAtK metric requires `--pass_at_k` argument to be set. Please set it to a valid integer."
                    )

            s = set(s)
            if dataset_cls.banned_subsets:
                if isinstance(dataset_cls.banned_subsets, str):
                    logger.warning(f"{dataset_cls}.banned_subsets should be a list of strings, not a string.")
                    banned_subsets = {dataset_cls.banned_subsets}  # type: ignore
                else:
                    banned_subsets = set(dataset_cls.banned_subsets)
                s -= banned_subsets

            available_subsets.update(s)
            available_subsets_by_cls.append(s)
    else:
        available_subsets_by_cls = [set() for _ in dataset_classes]

    # for wmt, en-xx and xx-en are both supported
    if "wmt" in dataset_name:  # matches "wmt16", "wmt17", ...
        for subset in available_subsets.copy():
            if subset.endswith("-en"):
                available_subsets.add("en-" + subset.split("-")[0])

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
    available_subsets_by_cls = get_subsets(dataset_name, dataset_classes, args, offline=args.dataset_path is not None)

    for dataset_cls, available_subsets in zip(dataset_classes, available_subsets_by_cls):

        cmd_subset_names = get_cmd_subset_names(args.subset_names, dataset_cls)
        if len(args.subset_names) > 0 and len(cmd_subset_names) == 0:
            continue

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
            subset_names = sorted(subset_names)
            max_eles = 5
            if len(subset_names) > max_eles or logger.level <= logging.DEBUG:
                subset_repr = ",".join(subset_names[:max_eles]) + " ..."
            else:
                subset_repr = ",".join(subset_names)
            logger.info("Loading dataset `%s` with subset(s): %s", dataset_name, subset_repr)
            if threading and len(subset_names) > 2:
                first_dataset = subset_names.pop(0)
                first_dataset = (
                    dataset_name + ":" + first_dataset, dataset_cls(dataset_name, args, model, first_dataset)
                )
                logger.info(f"Loading remaining subsets ...")
                logging.disable(logging.INFO)
                with ThreadPoolExecutor(max_workers=len(subset_names)) as executor:
                    res = [
                        executor.submit(
                            lambda s: (dataset_name + ":" + s, dataset_cls(dataset_name, args, model, s)), s
                        ) for s in subset_names
                    ]
                datasets = dict([first_dataset] + [f.result() for f in as_completed(res)])
                logging.disable(logging.NOTSET)
            else:
                datasets = {dataset_name + ":" + s: dataset_cls(dataset_name, args, model, s) for s in subset_names}
            yield datasets

        elif len(subset_names) == 1 and len(available_subsets) != 1 and accepts_subset(
            dataset_cls.load_args, overwrite_subset=len(cmd_subset_names) > 0, subset=next(iter(subset_names))
        ):
            # in some cases of get_dataset_config_names() have only one subset, loading dataset with the a subset name is not allowed in huggingface datasets library
            # len(available_subsets) == 0 means a special case, like wmt
            # race:middle (one of the subsets), coqa (default)
            subset_name = next(iter(subset_names))
            logger.info(f"Loading subset of dataset `{dataset_name}:{subset_name}`")
            yield {dataset_name + ":" + subset_name: dataset_cls(dataset_name, args, model, subset_name)}

        else:
            # copa (super_glue:copa) or anli
            logger.info(f"Loading dataset `{dataset_name}`")
            yield {dataset_name: dataset_cls(dataset_name, args, model)}


@catch_error
def load_datasets(args: "DatasetArguments", model: "Model") -> DatasetCollection:

    if model.model_backend == "vllm":
        args.batch_size = -1
        logger.info("Setting batch_size to -1, since vllm can automatically planning the optimal batch and order.")

    if model.args.prefix_caching and not model.is_local_model():
        logger.warning(
            "Prefix caching is only available for HuggingFaceModel. Automatically set prefix_caching to False"
        )
        model.args.prefix_caching = False

    datasets = []
    for d in args.dataset_names:
        datasets.extend(load_dataset(d, args, model, args.dataset_threading))
    datasets = {k: v for d in datasets for k, v in d.items()}
    if len(datasets) <= 0:
        raise ValueError("No datasets loaded.")
    dataset_collection = DatasetCollection(datasets)
    logger.debug(f"Evaluation datasets: {dataset_collection}")
    return dataset_collection
