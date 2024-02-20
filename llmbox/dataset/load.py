import importlib
import inspect
from logging import getLogger
from typing import Union

from datasets import get_dataset_config_names

from ..model.model import Model
from ..utils import DatasetArguments
from .dataset import Dataset, DatasetCollection
from .utils import accepts_subset

logger = getLogger(__name__)


def import_dataset_class(dataset_name: str) -> Dataset:
    if "wmt" in dataset_name:
        from .translation import Translation

        return Translation

    if 'squad' in dataset_name:
        from .squad import Squad

        return Squad

    module_path = __package__ + "." + dataset_name
    module = importlib.import_module(module_path)
    clsmembers = inspect.getmembers(module, inspect.isclass)

    for name, obj in clsmembers:
        if issubclass(obj, Dataset) and name.lower() == dataset_name.lower():
            logger.debug(f"Dataset class `{name}` imported from `{module_path}`.")
            return obj

    raise ValueError(
        f"Cannot find dataset class with name {dataset_name} in module {module_path}. "
        "Make sure the dataset class defines `name` attribute properly."
    )


def load_dataset(args: DatasetArguments, model: Model) -> Union[Dataset, DatasetCollection]:
    r"""Load corresponding dataset class.

    Args:
        args (Namespace): The global configurations.
        model (Model): Our class for model.

    Returns:
        Dataset: Our class for dataset.
    """
    dataset_cls = import_dataset_class(args.dataset_name)

    name = dataset_cls.load_args[0] if len(dataset_cls.load_args) > 0 else args.dataset_name
    try:
        available_subsets = set(get_dataset_config_names(name))
    except Exception as e:
        logger.info(f"Failed when trying to get_dataset_config_names: {e}")
        available_subsets = set()
    # TODO catch connection warning
    if available_subsets == {"default"}:
        available_subsets = set()
    logger.debug(f"{name} - available_subsets: {available_subsets}, load_args: {dataset_cls.load_args}")

    # for wmt, en-xx and xx-en are both supported
    if "wmt" in args.dataset_name:
        for subset in available_subsets.copy():
            available_subsets.add("en-" + subset.split("-")[0])

    # for mmlu and race dataset, remove "all" subset by default
    if args.dataset_name in {"mmlu", "race"} and len(args.subset_names) == 0:
        available_subsets.remove("all")

    # if dataset not in huggingface, allow to manually specify subset_names
    if len(available_subsets) and not available_subsets.issuperset(args.subset_names):
        raise ValueError(
            f"Specified subset names {args.subset_names} are not available. Available ones of {args.dataset_name} are: {available_subsets}"
        )

    # use specified subset_names if available
    subset_names = args.subset_names or available_subsets

    # load dataset
    if "squad_v2" in args.dataset_name:
        dataset_cls.load_args = ("squad_v2",)

    if len(subset_names) > 1 and accepts_subset(dataset_cls.load_args, overwrite_subset=len(args.subset_names) > 0):
        # race:middle,high (several subsets) , super_glue (all the subsets)
        logger.info(f"Loading subsets of dataset `{args.dataset_name}`: " + ", ".join(subset_names))
        datasets = {s: dataset_cls(args, model, s) for s in sorted(subset_names)}
        return DatasetCollection(datasets)
    elif len(subset_names) == 1 and len(available_subsets) != 1 and accepts_subset(
        dataset_cls.load_args, overwrite_subset=len(args.subset_names) > 0, subset=next(iter(subset_names))
    ):
        # in some cases of get_dataset_config_names() have only one subset, loading dataset with the a subset name is not allowed in huggingface datasets library
        # len(available_subsets) == 0 means a special case, like wmt
        # race:middle (one of the subsets), coqa (default)
        logger.info(f"Loading subset of dataset `{args.dataset_name}:{next(iter(subset_names))}`")
        return dataset_cls(args, model, next(iter(subset_names)))
    else:
        # copa (super_glue:copa) or mmlu
        logger.info(f"Loading dataset `{args.dataset_name}`")
        return dataset_cls(args, model)
