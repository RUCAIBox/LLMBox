import importlib
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import getLogger
from typing import TYPE_CHECKING, Union

from datasets import DownloadConfig, get_dataset_config_names

from ..metric import GPTEval
from .dataset import Dataset, DatasetCollection
from .utils import accepts_subset

if TYPE_CHECKING:
    # solve the circular import
    from ..model.model import Model
    from ..utils import DatasetArguments

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


def load_dataset(args: "DatasetArguments", model: "Model", threading: bool = True) -> Union[Dataset, DatasetCollection]:
    r"""Load corresponding dataset class.

    Args:
        args (Namespace): The global configurations.
        model (Model): Our class for model.

    Returns:
        Dataset: Our class for dataset.
    """
    try:
        dataset_cls = import_dataset_class(args.dataset_name)
    except ModuleNotFoundError:
        dataset_cls = import_dataset_class(next(iter(args.subset_names)))

    name = dataset_cls.load_args[0] if len(dataset_cls.load_args) > 0 else args.dataset_name
    download_config = DownloadConfig(use_etag=False)
    if args.dataset_path is None:
        try:
            available_subsets = set(
                get_dataset_config_names(name, download_config=download_config, trust_remote_code=True)
            )
        except Exception as e:
            logger.info(f"Failed when trying to get_dataset_config_names: {e}")
            available_subsets = set()
    else:
        available_subsets = set()

    if available_subsets == {"default"}:
        available_subsets = set()

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
    logger.debug(
        f"{name} - available_subsets: {available_subsets}, load_args: {dataset_cls.load_args}, final subset_names: {subset_names}"
    )

    # GPTEval requires openai-gpt
    if any(isinstance(m, GPTEval) for m in dataset_cls.metrics) and model.args.openai_api_key is None:
        raise ValueError(
            "OpenAI API key is required for GPTEval metrics. Please set it by passing a `--openai_api_key` or through environment variable `OPENAI_API_KEY`."
        )

    # load dataset
    if "squad_v2" in args.dataset_name:
        dataset_cls.load_args = ("squad_v2",)

    if len(subset_names) > 1 and accepts_subset(dataset_cls.load_args, overwrite_subset=len(args.subset_names) > 0):
        # race:middle,high (several subsets) , super_glue (all the subsets)
        logger.info(f"Loading subsets of dataset `{args.dataset_name}`: " + ", ".join(subset_names))
        if threading and len(subset_names) >= 8:
            with ThreadPoolExecutor(max_workers=len(subset_names)) as executor:
                res = [executor.submit(lambda s: (s, dataset_cls(args, model, s)), s) for s in subset_names]
            datasets = dict(sorted((f.result() for f in as_completed(res)), key=lambda x: x[0]))
        else:
            datasets = {s: dataset_cls(args, model, s) for s in sorted(subset_names)}
        dataset_collection = DatasetCollection(datasets)
        logger.debug(dataset_collection)
        return dataset_collection
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
