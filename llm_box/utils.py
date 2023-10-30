import argparse
import importlib
import inspect
from typing import Type, TypeVar, Callable, Optional
from argparse import Namespace
import coloredlogs

import torch

T = TypeVar('T')


@property
def NotImplementedField(self):
    raise NotImplementedError(f"{self.__class__.__name__} has not implemented field.")


def import_main_class(module_path, main_cls_type: Type[T], package: Optional[str] = None, filter: Callable[[Type[T]], bool]=None) -> Type[T]:
    """Import a module at module_path and return its main class, a Metric by default"""
    module = importlib.import_module(module_path, package)
    if filter is None:
        filter = lambda obj: True

    # Find the main class in our imported module
    module_main_cls = None
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, main_cls_type):
            if inspect.isabstract(obj) or not filter(obj):
                continue
            module_main_cls = obj
            break

    if module_main_cls is None:
        raise ValueError(f'Cannot find a class in {module_path} that is a subclass of {main_cls_type}')
    return module_main_cls


def _read_args_to_dict(args, results, name, value):
    results[name] = getattr(args, name, value)


def args_to_tokenizer_kwargs(args: Namespace):
    kwargs = dict()
    _read_args_to_dict(args, kwargs, "use_fast", False)
    _read_args_to_dict(args, kwargs, "padding_side", "left")
    return kwargs


def args_to_model_kwargs(args: Namespace):
    kwargs = dict()
    # `model.half()` is equivalent to `model.to(torch.float16)`
    torch_dtype = torch.float16 if getattr(args, "load_in_half", True) else torch.float32
    _read_args_to_dict(args, kwargs, "torch_dtype", torch_dtype)
    _read_args_to_dict(args, kwargs, "device_map", "auto")
    _read_args_to_dict(args, kwargs, "trust_remote_code", True)
    _read_args_to_dict(args, kwargs, "load_in_8bit", False)
    return kwargs


def parse_argument():
    r"""Parse arguments from command line. Using `argparse` for predefined ones, and an easy mannal parser for others (saved in `kwargs`).

    Returns:
        Namespace: the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="curie", help="The model name, e.g., cuire, llama")
    parser.add_argument("-d", "--dataset", type=str, default="copa", help="The model name, e.g., copa, gsm")
    parser.add_argument("-bsz", "--batch_size", type=int, default=1, help="The evaluation batch size")
    parser.add_argument("--evaluation_set", type=str, default=None, help="The set name for evaluation")
    parser.add_argument("--seed", type=int, default=2023, help="The random seed")
    parser.add_argument("-inst", "--instruction", type=str, default="", help="The instruction to format each instance")
    parser.add_argument("--example_set", type=str, default=None, help="The set name for demonstration")
    parser.add_argument("-shots", "--num_shots", type=int, default=0, help="The few-shot number for demonstration")
    parser.add_argument("--max_example_tokens", type=int, default=1024, help="The maximum token number of demonstration")
    parser.add_argument("--example_separator_string", type=str, default="\n\n", help="The string to separate each demonstration")
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

    coloredlogs.install("DEBUG")

    return args
