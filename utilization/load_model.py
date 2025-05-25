from functools import wraps
from logging import getLogger
from typing import TYPE_CHECKING

from .utils.catch_error import catch_error

if TYPE_CHECKING:
    # solve the circular import
    from .model import Model
    from .utils import ModelArguments

logger = getLogger(__name__)

__all__ = ["register_model", "load_model"]

LOAD_REGISTERY = {}


def register_model(model_backend):

    def inner_decrator(fn):
        LOAD_REGISTERY[model_backend] = fn
        return fn

    return inner_decrator


@register_model(model_backend="openai")
def load_openai(args: "ModelArguments"):
    logger.info(f"Loading OpenAI API model `{args.model_name_or_path}`.")
    from .model.openai_model import Openai

    return Openai(args)


@register_model(model_backend="anthropic")
def load_anthropic(args: "ModelArguments"):
    logger.info(f"Loading Anthropic API model `{args.model_name_or_path}`.")
    from .model.anthropic_model import Anthropic

    return Anthropic(args)


@register_model(model_backend="dashscope")
def load_dashscope(args: "ModelArguments"):
    logger.info(f"Loading Dashscope (Aliyun) API model `{args.model_name_or_path}`.")
    from .model.dashscope_model import Dashscope

    return Dashscope(args)


@register_model(model_backend="qianfan")
def load_qianfan(args: "ModelArguments"):
    logger.info(f"Loading Qianfan (Baidu) API model `{args.model_name_or_path}`.")
    from .model.qianfan_model import Qianfan

    return Qianfan(args)


@register_model(model_backend="vllm")
def load_vllm(args: "ModelArguments"):
    try:
        import vllm
        logger.debug(f"vllm version: {vllm.__version__}")

        from .model.vllm_model import vllmModel

        return vllmModel(args)
    except ModuleNotFoundError:
        logger.warning(f"vllm has not been installed, falling back.")
        return None
    except ValueError as e:
        if "are not supported for now" in str(e):
            logger.warning(f"vllm has not supported the architecture of {args.model_name_or_path} for now.")
            return None
        elif "divisible by tensor parallel size" in str(e):
            raise ValueError(f"Set an appropriate tensor parallel size via CUDA_VISIBLE_DEVICES: {e}")
        else:
            raise e


@register_model(model_backend="huggingface")
def load_huggingface(args: "ModelArguments"):
    logger.info(f"Loading HuggingFace model `{args.model_name_or_path}`.")
    from .model.huggingface_model import HuggingFaceModel

    return HuggingFaceModel(args)


@register_model(model_backend="megatron")
def load_megatron(args: "ModelArguments"):
    logger.info(f"Loadding Megatron model `{args.model_name_or_path}` from {args.megatron_path}.")
    from .model.megatron_model import MegatronModel

    return MegatronModel(args)


@catch_error()
def load_model(args: "ModelArguments") -> "Model":
    r"""Load corresponding model class.

    Args:
        args (ModelArguments): The global configurations.

    Returns:
        Model: Our class for model.
    """
    loads = args.model_backend
    if loads not in LOAD_REGISTERY:
        raise ValueError(f"Model backend `{loads}` is not supported.")

    if loads == "vllm":
        loads = ["vllm", "huggingface"]
    else:
        loads = [loads]

    for load in loads:
        model = LOAD_REGISTERY[load](args)
        if model is not None:
            return model

    raise ValueError(f"Model backend `{loads}` is not supported.")
