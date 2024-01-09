from logging import getLogger

from ..utils import ModelArguments
from .huggingface_model import HuggingFaceModel
from .model import Model
from .openai import Openai
from .enum import OPENAI_MODELS

logger = getLogger(__name__)


def load_model(args: ModelArguments) -> Model:
    r"""Load corresponding model class.

    Args:
        args (Namespace): The global configurations.

    Returns:
        Model: Our class for model.
    """
    if args.model_name_or_path.lower() in OPENAI_MODELS:
        logger.info(f"Loading OpenAI API model `{args.model_name_or_path.lower()}`.")
        return Openai(args)
    else:
        logger.info(f"Loading HuggingFace Transformers model `{args.model_name_or_path}`.")
        return HuggingFaceModel(args.model_name_or_path, args)
