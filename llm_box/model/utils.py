import importlib
from logging import getLogger

logger = getLogger(__name__)

OPENAI_MODELS = ['ada', 'babbage', 'curie', 'davinci', 'babbage-002', 'davinci-002', 'gpt-3.5-turbo', "gpt-3.5-turbo-instruct", "text-davinci-003"]


def load_model(args):
    r"""Load corresponding model class.

    Args:
        args (ModelArguments): The global configurations.

    Returns:
        Model: Our class for model.
    """
    if args.model_name_or_path.lower() in OPENAI_MODELS:
        logger.info(f"Loading OpenAI API model `{args.model_name_or_path.lower()}`.")
        from .openai import Openai
        model = Openai(args)
    else:
        logger.info(f"Loading HuggingFace pretrained model `{args.model_name_or_path}`.")
        model = importlib.import_module(f".{args.model_name_or_path}")
        model = getattr(model, args.model_name_or_path)(args)
    return model
