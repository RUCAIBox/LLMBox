import argparse
import importlib
import inspect
from typing import Type, TypeVar

T = TypeVar('T')


def import_main_class(module_path, main_cls_type: Type[T]) -> Type[T]:
    """Import a module at module_path and return its main class, a Metric by default"""
    module = importlib.import_module(module_path)

    # Find the main class in our imported module
    module_main_cls = None
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, main_cls_type):
            if inspect.isabstract(obj):
                continue
            module_main_cls = obj
            break

    if module_main_cls is None:
        raise ValueError(f'Cannot find a class in {module_path} that is a subclass of {main_cls_type}')
    return module_main_cls


def parse_argument():
    r"""Parse arguments from command line. Using `argparse` for predefined ones, and an easy mannal parser for others (saved in `kwargs`).

    Returns:
        Namespace: the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="curie", help="The model name, e.g., cuire, llama")
    parser.add_argument("-d", "--dataset", type=str, default="copa", help="The model name, e.g., copa, gsm")
    parser.add_argument("-bsz", "--batch_size", type=int, default=1, help="The evaluation batch size")
    parser.add_argument("--evaluation_set", type=str, default="validation", help="The set name for evaluation")
    parser.add_argument("--seed", type=int, default=2023, help="The random seed")
    parser.add_argument("-sys", "--system_prompt", type=str, default="", help="The system prompt of the model")
    parser.add_argument(
        "-format",
        "--instance_format",
        type=str,
        default="{source}{target}",
        help="The format to format the `source` and `target` for each instance"
    )
    parser.add_argument("--example_set", type=str, default="train", help="The set name for demonstration")
    parser.add_argument("-shots", "--num_shots", type=int, default=0, help="The few-shot number for demonstration")
    parser.add_argument(
        "--max_example_tokens", type=int, default=1024, help="The maximum token number of demonstration"
    )
    parser.add_argument("-api", "--openai_api_key", type=str, default="", help="The OpenAI API key")

    args, unparsed = parser.parse_known_args()

    new_unparsed = []
    for arg in unparsed:
        if arg.find('=') >= 0:
            new_unparsed.append(arg.split('='))
        else:
            new_unparsed.append(arg)

    assert len(new_unparsed) % 2 == 0, "Arguments parsing error!"
    kwargs = {}
    for i in range(len(new_unparsed) // 2):
        key, value = new_unparsed[i * 2:i * 2 + 2]
        if key.find('--') != 0:
            raise KeyError
        else:
            key = key[2:]
            try:
                value = eval(value)
            except:
                pass
            setattr(args, key, value)
            kwargs[key] = value
    args.kwargs = kwargs

    return args
