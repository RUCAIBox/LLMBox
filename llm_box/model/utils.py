import importlib
import re

OPENAI_MODELS = ['ada', 'babbage', 'curie', 'davinci', 'babbage-002', 'davinci-002', 'gpt-3.5-turbo']


def load_model(args):
    r"""Load corresponding model class.

    Args:
        args (ModelArguments): The global configurations.

    Returns:
        Model: Our class for model.
    """
    if args.model_name_or_path.lower() in OPENAI_MODELS:
        from .openai import Openai
        model = Openai(args)
    else:
        model = importlib.import_module(f".{args.model_name_or_path}")
        model = getattr(model, args.model_name_or_path)(args)
    return model
