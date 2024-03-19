from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # solve the circular import
    from ..utils import ModelArguments
    from .model import Model

logger = getLogger(__name__)


def load_model(args: "ModelArguments") -> "Model":
    r"""Load corresponding model class.

    Args:
        args (ModelArguments): The global configurations.

    Returns:
        Model: Our class for model.
    """
    if args.is_openai_model():
        logger.info(f"Loading OpenAI API model `{args.model_name_or_path.lower()}`.")
        from .openai import Openai

        return Openai(args)
    elif args.is_anthropic_model():
        logger.info(f"Loading Anthropic API model `{args.model_name_or_path.lower()}`.")
        from .anthropic import Anthropic

        return Anthropic(args)
    elif args.is_dashscope_model():
        logger.info(f"Loading Dashscope (Aliyun) API model `{args.model_name_or_path.lower()}`.")
        from .dashscope import Dashscope

        return Dashscope(args)
    elif args.is_qianfan_model():
        logger.info(f"Loading Qianfan (Baidu) API model `{args.model_name_or_path.lower()}`.")
        from .qianfan import Qianfan

        return Qianfan(args)
    else:
        if args.vllm:
            try:
                import vllm

                from .vllm_model import vllmModel

                return vllmModel(args)
            except ModuleNotFoundError:
                args.vllm = False
                logger.warning(f"vllm has not been installed, falling back to huggingface.")
            except ValueError as e:
                if "are not supported for now" in str(e):
                    args.vllm = False
                    logger.warning(f"vllm has not supported the architecture of {args.model_name_or_path} for now.")
                elif "divisible by tensor parallel size" in str(e):
                    raise ValueError(f"Set an appropriate tensor parallel size via CUDA_VISIBLE_DEVICES: {e}")
                else:
                    raise e
        from .huggingface_model import HuggingFaceModel

        return HuggingFaceModel(args)
