from copy import deepcopy
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

if TYPE_CHECKING:
    from .arguments import ModelArguments

logger = getLogger(__name__)

Val = Any
ValMap = Callable[[Val, "GenerationArg"], Dict[str, Val]]


@dataclass
class GenerationArg:

    default: Optional[Any]
    _type: Optional[type]
    transform_value: Optional[Callable[[Any], Any]]
    transform_key: Optional[str]
    nullable: bool
    needs: Optional[Union[dict, ValMap]]
    extra_body: bool


def generation_arg(
    *,
    default: Optional[Any] = None,
    _type: Optional[type] = None,
    transform_value: Optional[Callable[[Any], Any]] = None,
    transform_key: Optional[str] = None,
    nullable: bool = False,
    needs: Optional[Union[dict, ValMap]] = None,
    extra_body: bool = False,
) -> GenerationArg:
    assert _type is None or transform_value is None, "Cannot have both _type and transform_value"
    return GenerationArg(default, _type, transform_value, transform_key, nullable, needs, extra_body)


def set_args(
    generation_kwargs: Dict[str, Val],
    key: str,
    value: Val,
    details: Optional[GenerationArg] = None,
    extra_body=False,
):

    if details is not None:

        # type casting
        if details._type is not None and value is not None:
            value = details._type(value)

        # transform
        if details.transform_value is not None and value is not None:
            value = details.transform_value(value)

        extra_body = details.extra_body

    display_key = f"extra_body.{key}" if extra_body else key
    logger.debug(f"Setting {display_key} to {value}")
    if extra_body:
        if "extra_body" not in generation_kwargs:
            generation_kwargs["extra_body"] = {}

        if key in generation_kwargs["extra_body"] and generation_kwargs["extra_body"][key] != value:
            raise ValueError(f"Conflict value for {key}: {generation_kwargs['extra_body'][key]} vs {value}")

        generation_kwargs["extra_body"][key] = value
    else:

        if key in generation_kwargs and generation_kwargs[key] != value:
            raise ValueError(f"Conflict value for {key}: {generation_kwargs[key]} vs {value}")

        generation_kwargs[key] = value


def resolve_generation_args(
    model_args: "ModelArguments",
    extra_model_args: Dict[str, Any],
    endpoint_schema: Dict[str, GenerationArg],
    extra_generation_args: Optional[Dict[str, Union[Val, ValMap]]] = None,
) -> Dict[str, Any]:
    generation_kwargs = {}
    if extra_generation_args is None:
        extra_generation_args = {}

    for key, details in deepcopy(endpoint_schema).items():
        # ModelArguments (cmd) > extra_model_args > ModelArguments (default)
        if not model_args.passed_in_commandline(key):
            value = extra_model_args.pop(key, None)
        else:
            value = None
        if value is None:
            value = getattr(model_args, key, None)

        # overrides
        if key in extra_generation_args:
            extra = extra_generation_args.pop(key)
            if value is None and not details.nullable:
                continue
            if callable(extra):
                overrided = extra(value, details)
                for new_key, new_value in overrided.items():
                    set_args(generation_kwargs, new_key, new_value, details)
            else:
                set_args(generation_kwargs, key, extra, details)
            continue

        # set default values
        if value is None and details.default is not None:
            value = details.default

        # set alias after default values
        if details.transform_key is not None:
            key = details.transform_key

        # skip if no value
        if value is None and not details.nullable:
            continue

        if isinstance(details.needs, dict):
            for need, need_value in details.needs.items():
                set_args(generation_kwargs, need, need_value, endpoint_schema.get(need, None))
        elif callable(details.needs):
            need_dict = details.needs(value, model_args)
            for need, need_value in need_dict.items():
                set_args(generation_kwargs, need, need_value, endpoint_schema.get(need, None))

        set_args(generation_kwargs, key, value, details)

    if extra_generation_args:
        if any(callable(v) for v in extra_generation_args.values()):
            raise ValueError("Extra model args must be resolved before this point")
        generation_kwargs.update(extra_generation_args)
        extra_generation_args.clear()

    return generation_kwargs
