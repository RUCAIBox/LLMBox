import importlib
import warnings
from logging import getLogger

logger = getLogger(__name__)

OPENAI_MODELS = ['ada', 'babbage', 'curie', 'davinci', 'babbage-002', 'davinci-002', 'gpt-3.5-turbo']


def load_model(args, batch_size):
    r"""Load corresponding model class.

    Args:
        args (ModelArguments): The global configurations.
        batch_size (int): The batch size for model.

    Returns:
        Model: Our class for model.
    """
    if args.model_name_or_path.lower() in OPENAI_MODELS:
        logger.info(f"Loading OpenAI API model `{args.model_name_or_path.lower()}`.")
        from .openai import Openai
        if args.model_name_or_path.lower() == 'gpt-3.5-turbo' and batch_size > 1:
            args.batch_size = 1
            warnings.warn("gpt-3.5-turbo doesn't support batch_size > 1, automatically set batch_size=1.")
        model = Openai(args)
    else:
        logger.info(f"Loading HuggingFace pretrained model `{args.model_name_or_path}`.")
        model = importlib.import_module(f".{args.model_name_or_path}")
        model = getattr(model, args.model_name_or_path)(args)
    return model
