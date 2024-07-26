import difflib
import importlib
import inspect
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import zip_longest
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Type

from datasets import DownloadConfig, get_dataset_config_names

from .dataset import Dataset, DatasetCollection
from .dataset.dataset_utils.raw_dataset_loader import accepts_subset
from .dataset_enum import DATASET_ALIASES
from .metric import GPTEval, PassAtK
from .utils.catch_error import catch_error
from .utils.hfd import get_script_path, huggingface_download
from .utils.logging import list_datasets

if TYPE_CHECKING:
    # solve the circular import
    from .model import Model
    from .utils import DatasetArguments, EvaluationArguments

logger = getLogger(__name__)

__all__ = ["load_datasets", "register_dataset"]

ABSTRACT_DATASET = {"Dataset", "GenerationDataset", "MultipleChoiceDataset", "SquadDataset"}
REGISTERY = {}


def _validate_dataset_class(cls):
    name = cls.__name__
    return issubclass(cls, Dataset) and name not in ABSTRACT_DATASET


def _fuzzy_match_prompt(dataset_name) -> str:
    all_datasets = list_datasets()
    matches = difflib.get_close_matches(dataset_name, list(all_datasets), cutoff=0.6)
    if len(matches) == 0:
        fuzzy_match = f" Available choices are: {all_datasets}."
    else:
        fuzzy_match = f" Possible choices are: {matches}."
    return fuzzy_match


def _import_dataset_class(dataset_name: str) -> Type[Dataset]:

    module_path = __package__ + ".dataset." + dataset_name
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        fuzzy_match = _fuzzy_match_prompt(dataset_name) if f"utilization.dataset.{dataset_name}" in str(e) else ""
        raise ValueError(f"Invalid dataset: {dataset_name}.{fuzzy_match}\n{e}") from e
    clsmembers = inspect.getmembers(module, inspect.isclass)

    for name, obj in clsmembers:
        if _validate_dataset_class(obj):
            logger.debug(f"Dataset class `{name}` imported from `{module_path}`.")
            return obj

    raise ValueError(
        f"Cannot find dataset class with name {dataset_name} in module {module_path}. "
        "Make sure the dataset class defines `name` attribute properly."
    )


def import_dataset_classes(dataset_name: str) -> List[Type[Dataset]]:
    """Import dataset classes from the dataset_name. Look up order:

    1. Registered datasets with `register_dataset`
    2. Dataset aliases defined in `DATASET_ALIASES`
    3. Native dataset classes in `utilization.dataset.{dataset_name}`
    """

    if dataset_name in REGISTERY:
        return [REGISTERY[dataset_name]]
    elif dataset_name in DATASET_ALIASES:
        logger.info("Loading dataset aliases: %s -> %s", dataset_name, DATASET_ALIASES[dataset_name])
        return [_import_dataset_class(alias) for alias in DATASET_ALIASES[dataset_name]]
    else:
        return [_import_dataset_class(dataset_name)]


def get_subsets(
    dataset_name: str,
    dataset_classes: List[Type[Dataset]],
    args: "DatasetArguments",
    cache_paths: List[Optional[str]],
) -> List[Set[str]]:

    available_subsets = set()
    available_subsets_by_cls: List[Set[str]] = []

    for dataset_cls, cache_path in zip_longest(dataset_classes, cache_paths):

        if dataset_cls.load_args is None:
            available_subsets_by_cls.append(set())
            continue

        # dynamically set load_args for wmt datasets, in order to support wmt series datasets
        if not dataset_cls.load_args:
            dataset_cls.load_args = (dataset_name,)

        download_config = DownloadConfig(use_etag=False)
        paths = [cache_path, args.dataset_path, dataset_cls.load_args[0]]
        if args.dataset_path is not None:
            paths = [str(get_script_path(cache_path))] + paths
        if cache_path is not None:
            paths = [str(get_script_path(cache_path))] + paths

        found_config = False
        for path in paths:
            if path is None:
                continue

            try:
                s = get_dataset_config_names(path=path, download_config=download_config, trust_remote_code=True)
                found_config = True
                break
            except Exception as e:
                logger.info(f"Failed when trying to get_dataset_config_names({path}): {e}. Trying another method...")

        logger.debug(f"get_dataset_config_names({path}): {s}")

        if not found_config:
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            s = []

        if s == ["default"]:
            s = []

        for m in getattr(dataset_cls, "metrics", []):
            if isinstance(m, GPTEval):
                import openai
                if openai.api_key is None:
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
            logger.debug(f"Removing banned subsets {banned_subsets} of {dataset_cls} from available subsets.")
            s -= banned_subsets

        available_subsets.update(s)
        available_subsets_by_cls.append(s)

    # for wmt, en-xx and xx-en are both supported
    if "wmt" in dataset_name:  # matches "wmt16", "wmt17", ...
        for subset in available_subsets.copy():
            if subset.endswith("-en"):
                available_subsets.add("en-" + subset.split("-")[0])

    for idx, a in enumerate(available_subsets_by_cls):
        available_subsets_by_cls[idx] = a.intersection(available_subsets)
        available_subsets -= a
        logger.debug(f"Available subsets of {dataset_classes[idx].__name__}: {available_subsets_by_cls[idx]}")

    return available_subsets_by_cls


def expand_cmd_subset_names(cmd_subset_names: Set[str], dataset_classes: List[Type["Dataset"]]) -> Set[str]:
    """Get the subset names from the command line arguments. If the subset name is a category, expand it to the actual subset names."""
    results = set()
    all_categorized_subsets = dict()
    for cls in dataset_classes:
        if cls.categorized_subsets is not None:
            all_categorized_subsets.update(cls.categorized_subsets)

    for cmd_subset in cmd_subset_names:
        if cmd_subset.startswith("[") and cmd_subset.endswith("]"):
            cat_name = cmd_subset[1:-1]
            if cat_name not in all_categorized_subsets:
                all_classes = ", ".join(d.__name__ for d in dataset_classes)
                raise ValueError(f"Category `{cat_name}` not found in dataset:", all_classes)
            categorized_subsets = all_categorized_subsets[cat_name]
            logger.info(f"Expanding category `{cmd_subset}` to `{categorized_subsets}`")
            results.update(categorized_subsets)
        else:
            results.add(cmd_subset)
    return results


def load_dataset(
    dataset_name: str,
    args: "DatasetArguments",
    model: "Model",
    evaluation_args: "EvaluationArguments",
    evaluation_data: Optional[List[Dict[str, Any]]] = None,
    example_data: Optional[List[Dict[str, Any]]] = None,
) -> Iterator[Dict[str, Dataset]]:
    """Load corresponding dataset class. One dataset class contains one subset,
    e.g., Mmlu(abstract_algebra), Mmlu()

    1. Load dataset classes from dataset_name, e.g. `agieval` -> `Agieval_cot`
    and `Agieval_single_choice`
    2. Get available subsets for each dataset class, e.g., `Agieval_cot` ->
    `['lsat-ar', ...]`, `Agieval_single_choice` -> `[logiqa-zh', ...]`
    3. Get subset names from command line arguments and get the intersection.
    4. Instantiate each dataset class with corresponding subset name.

    Args:
        - dataset_name (str): The name of the dataset.
        - args (DatasetArguments): The global configurations.
            - subset_names
            - dataset_path
            - hf_mirror
            - hfd_cache_path
        - model (Model): Our class for model.
        - evaluation_args (EvaluationArguments): The evaluation arguments
            - dataset_threading
        - evaluation_data (Optional[List[Dict[str, Any]]]): The evaluation data for the dataset.
        -

    Returns:
        An iterator of dictionaries grouped by dataset classes, each containing a mapping of display_names to dataset instances.
    """

    # get the subset names
    if ":" in dataset_name:
        dataset_name, cmd_subset_names = dataset_name.split(":")
        cmd_subset_names = set(cmd_subset_names.split(","))
    else:
        cmd_subset_names = set()

    dataset_classes = import_dataset_classes(dataset_name)
    cache_paths = []
    for dcls in dataset_classes:
        if dcls.load_args is None:
            continue
        elif len(dcls.load_args) > 0:
            cache_paths.append(
                huggingface_download(
                    dcls.load_args[0],
                    args.hfd_cache_path,
                    hf_username=evaluation_args.hf_username,
                    hf_token=evaluation_args.hf_token,
                    mirror=args.hf_mirror
                )
            )
        else:
            # dynamically set load_args for wmt datasets, in order to support wmt series datasets
            cache_paths.append(
                huggingface_download(
                    dataset_name,
                    args.hfd_cache_path,
                    hf_username=evaluation_args.hf_username,
                    hf_token=evaluation_args.hf_token,
                    mirror=args.hf_mirror
                )
            )
    available_subsets_by_cls = get_subsets(dataset_name, dataset_classes, args, cache_paths)
    expanded_cmd_subset_names = expand_cmd_subset_names(cmd_subset_names, dataset_classes)

    for dataset_cls, available_subsets, cache_path in zip_longest(
        dataset_classes, available_subsets_by_cls, cache_paths
    ):

        if not args.passed_in_commandline("dataset_path"):
            args.dataset_path = cache_path

        if len(cmd_subset_names) > 0 and len(expanded_cmd_subset_names) == 0:
            continue

        # if dataset not in huggingface, allow to manually specify subset_names
        if len(available_subsets) and not available_subsets.issuperset(expanded_cmd_subset_names):
            na = expanded_cmd_subset_names - available_subsets
            raise ValueError(
                f"Specified subset names {na} are not available. Available ones of {dataset_name} are: {available_subsets}"
            )

        # use specified subset_names if available
        cmd_subset_names = expanded_cmd_subset_names or available_subsets
        logger.debug(
            f"{dataset_name} - available_subsets: {available_subsets}, load_args: {dataset_cls.load_args}, final subset_names: {cmd_subset_names}"
        )

        if len(cmd_subset_names
               ) > 1 and accepts_subset(dataset_cls.load_args, overwrite_subset=len(expanded_cmd_subset_names) > 0):
            # Example: race:middle,high (several subsets) , super_glue (all the subsets)

            # sort the subset names and log to terminal
            cmd_subset_names = sorted(cmd_subset_names)
            max_eles = 5
            if len(cmd_subset_names) > max_eles or logger.level <= logging.DEBUG:
                subset_repr = ",".join(cmd_subset_names[:max_eles]) + " ..."
            else:
                subset_repr = ",".join(cmd_subset_names)
            logger.info("Loading dataset `%s` with subset(s): %s", dataset_name, subset_repr)

            if evaluation_args.dataset_threading and len(cmd_subset_names) > 2:

                # load the first dataset in the main thread (only show the INFO log message for the first dataset)
                first_dataset = cmd_subset_names.pop(0)
                first_dataset = (
                    dataset_name + ":" + first_dataset,
                    dataset_cls(dataset_name, args, model, first_dataset, evaluation_data, example_data)
                )
                logger.info(f"Loading remaining subsets ...")
                logging.disable(logging.INFO)

                # load the remaining datasets in parallel
                with ThreadPoolExecutor(max_workers=len(cmd_subset_names)) as executor:
                    res = [
                        executor.submit(
                            lambda s: (
                                dataset_name + ":" + s,
                                dataset_cls(dataset_name, args, model, s, evaluation_data, example_data)
                            ), s
                        ) for s in cmd_subset_names
                    ]
                datasets = dict([first_dataset] + [f.result() for f in as_completed(res)])
                logging.disable(logging.NOTSET)
            else:
                # load all datasets one by one
                datasets = {
                    dataset_name + ":" + s: dataset_cls(dataset_name, args, model, s, evaluation_data, example_data)
                    for s in cmd_subset_names
                }
            yield datasets

        elif len(cmd_subset_names) == 1 and len(available_subsets) != 1 and accepts_subset(
            dataset_cls.load_args,
            overwrite_subset=len(expanded_cmd_subset_names) > 0,
            subset=next(iter(cmd_subset_names))
        ):
            # in some cases of get_dataset_config_names() have only one subset, loading dataset with the a subset name is not allowed in huggingface datasets library
            # len(available_subsets) == 0 means a special case, like wmt
            # race:middle (one of the subsets), coqa (default)
            subset_name = next(iter(cmd_subset_names))
            logger.info(f"Loading subset of dataset `{dataset_name}:{subset_name}`")
            yield {
                dataset_name + ":" + subset_name:
                dataset_cls(dataset_name, args, model, subset_name, evaluation_data, example_data)
            }

        else:
            # copa (super_glue:copa) or anli
            logger.info(f"Loading dataset `{dataset_name}`")
            yield {dataset_name: dataset_cls(dataset_name, args, model, None, evaluation_data, example_data)}

        expanded_cmd_subset_names.difference_update(cmd_subset_names)


@catch_error()
def load_datasets(
    args: "DatasetArguments",
    model: "Model",
    evaluation_args: "EvaluationArguments",
    evaluation_data: Optional[List[Dict[str, Any]]] = None,
    example_data: Optional[List[Dict[str, Any]]] = None,
) -> DatasetCollection:

    # batch size for vllm is set after model is loaded
    if model.is_vllm_model():
        args.batch_size = -1
        args.auto_batch_size = False
        logger.info("Setting batch_size to -1, since vllm can automatically planning the optimal batch and order.")

    # get all the dataset classes
    datasets = []
    for d in args.dataset_names:
        datasets.extend(
            load_dataset(
                d,
                args,
                model,
                evaluation_args,
                evaluation_data=evaluation_data,
                example_data=example_data,
            )
        )
    datasets = {k: v for d in datasets for k, v in d.items()}
    logger.debug(datasets)
    if len(datasets) <= 0:
        raise ValueError("No datasets loaded.")

    # collect all the datasets into a DatasetCollection
    dataset_collection = DatasetCollection(datasets)
    logger.debug(f"Evaluation datasets: {dataset_collection}")
    return dataset_collection
