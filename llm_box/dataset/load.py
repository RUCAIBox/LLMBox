import importlib
import inspect
from logging import getLogger
from typing import Union

from datasets import get_dataset_config_names

from ..model.model import Model
from ..prompt import PEMethod
from ..utils import DatasetArguments
from .dataset import Dataset, DatasetCollection

logger = getLogger(__name__)


def import_dataset_class(dataset_name: str) -> Dataset:
    module_path = __package__ + "." + dataset_name
    module = importlib.import_module(module_path)
    clsmembers = inspect.getmembers(module, inspect.isclass)

    for name, obj in clsmembers:
        if issubclass(obj, Dataset) and getattr(obj, "name", None) == dataset_name:
            logger.debug(f"Dataset class `{name}` imported from `{module_path}`.")
            return obj

    raise ValueError(
        f"Cannot find dataset class with name {dataset_name} in module {module_path}. "
        "Make sure the dataset class defines `name` attribute properly."
    )


def load_dataset(args: DatasetArguments, model: Model, method: PEMethod) -> Union[Dataset, DatasetCollection]:
    r"""Load corresponding dataset class.

    Args:
        args (Namespace): The global configurations.
        model (Model): Our class for model.

    Returns:
        Dataset: Our class for dataset.
    """

    # get available subset names and specified subset names
    try:
        assert args.dataset_path is None, "Load dataset from local path."
        available_subsets = set(get_dataset_config_names(args.dataset_name))
    except Exception as e:
        available_subsets = None
        logger.info(f"Receive {e.__class__.__name__}: {e}. Use `subset_names` instead.")

    if available_subsets is not None and not available_subsets.issuperset(args.subset_names):
        raise ValueError(
            f"Specified subset names {args.subset_names} are not available. Available ones are: {available_subsets}"
        )

    subset_names = args.subset_names or available_subsets or set()

    # load dataset
    dataset_cls = import_dataset_class(args.dataset_name)
    if len(subset_names) > 1 and len(dataset_cls.load_args) == 1:
        logger.info(f"Loading subsets of dataset `{args.dataset_name}`: " + ", ".join(subset_names))
        datasets = {s: dataset_cls(args, model, method, s) for s in subset_names}
        return DatasetCollection(datasets)
    elif len(subset_names) == 1 and len(dataset_cls.load_args) == 1:
        logger.info(f"Loading subset of dataset `{args.dataset_name}`: {next(iter(subset_names))}")
        return dataset_cls(args, model, method, next(iter(subset_names)))
    else:
        logger.info(f"Loading dataset `{args.dataset_name}`")
        return dataset_cls(args, model, method)
