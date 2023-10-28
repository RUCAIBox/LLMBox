import importlib

OPENAI_MODELS = ['ada', 'babbage', 'curie', 'davinci', 'babbage-002', 'davinci-002']


def load_model(args):
    r"""Load corresponding model class.

    Args:
        args (Namespace): The global configurations.

    Returns:
        Model: Our class for model.
    """
    if args.model.lower() in OPENAI_MODELS:
        from .openai import Openai
        model = Openai(args)
    else:
        model = importlib.import_module(f".{args.model}")
        model = getattr(model, args.model)(args)
    return model
