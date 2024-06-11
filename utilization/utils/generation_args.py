from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class GenerationArg:

    default: Optional[Any]
    _type: Optional[type]
    transform_value: Optional[Callable[[Any], Any]]
    transform_key: Optional[str]
    nullable: bool
    needs: Optional[dict]
    extra_body: bool


def generation_arg(
    *,
    default: Optional[Any] = None,
    _type: Optional[type] = None,
    transform_value: Optional[Callable[[Any], Any]] = None,
    transform_key: Optional[str] = None,
    nullable: bool = False,
    needs: Optional[dict] = None,
    extra_body: bool = False,
) -> GenerationArg:
    return GenerationArg(default, _type, transform_value, transform_key, nullable, needs, extra_body)
