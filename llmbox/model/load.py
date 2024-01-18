from ..utils import ModelArguments, getQueuedLogger
from .enum import OPENAI_MODELS
from .model import Model

logger = getQueuedLogger(__name__)


def load_model(args: ModelArguments) -> Model:
    r"""Load corresponding model class.

    Args:
        args (Namespace): The global configurations.

    Returns:
        Model: Our class for model.
    """
    if args.model_name_or_path.lower() in OPENAI_MODELS:
        logger.info(f"Loading OpenAI API model `{args.model_name_or_path.lower()}`.")
        args.vllm = False
        from .openai import Openai
        return Openai(args)
    else:
        if args.vllm:
            try:
                from .vllm_model import vllmModel
                return vllmModel(args)
            except ValueError as e:
                if 'are not supported for now' in str(e):
                    args.vllm = False
                    logger.warning(f"vllm has not supported the architecture of {args.model_name_or_path} for now.")
                else:
                    raise e
            except Exception as e:
                raise e
        from .huggingface_model import HuggingFaceModel
        return HuggingFaceModel(args)
