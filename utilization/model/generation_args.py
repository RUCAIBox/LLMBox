from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, TypeVar, Union


@dataclass
class GenerationArg:

    default: Optional[Any]
    _type: Optional[type]
    transform: Optional[Callable[[Any], Any]]
    alias: Optional[str]
    nullable: bool


def generation_arg(
    *,
    default: Optional[Any] = None,
    _type: Optional[type] = None,
    transform: Optional[Callable[[Any], Any]] = None,
    alias: Optional[str] = None,
    nullable: bool = False,
) -> GenerationArg:
    return GenerationArg(default, _type, transform, alias, nullable)
