from .openai import Openai
from .huggingface_model import HuggingFaceModel
from .utils import OPENAI_MODELS, load_tokenizer


def load_model(args):
    r"""Load corresponding model class.

    Args:
        args (Namespace): The global configurations.

    Returns:
        Model: Our class for model.
    """
    if args.model.lower() in OPENAI_MODELS:
        model = Openai(args)
    else:
        model = HuggingFaceModel(args.model, args)
    return model
