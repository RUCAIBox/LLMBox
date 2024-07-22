import sys

import pytest

sys.path.append('.')

from utilization import ModelArguments
from utilization.model.model import ApiModel, EnsureTypeError, ensure_type

from ..fixtures import *


class FakeModel(ApiModel):

    _raise_errors = ()

    @ensure_type(list)
    def _chat_completions(self, type_):

        if type_ is list:
            return [{"content": "choices 1"}, {"content": "choices 2"}]
        elif type_ is None:
            return None
        else:
            raise ValueError("Invalid type")

    @staticmethod
    @ensure_type(str)
    def _get_assistant(type_) -> str:
        if type_ is None:
            return None
        else:
            return "Hello"


def test_ensure_type_str():

    args = ModelArguments(model_name_or_path="gpt-3.5-turbo", openai_api_key="fake-key")
    model = FakeModel(args)

    with pytest.raises(EnsureTypeError):
        model._get_assistant(None)

    assert model._get_assistant(str) == "Hello"


def test_ensure_type_list():

    args = ModelArguments(model_name_or_path="gpt-3.5-turbo", openai_api_key="fake-key")
    model = FakeModel(args)

    with pytest.raises(EnsureTypeError):
        model._chat_completions(None)

    assert model._chat_completions(list) == [{"content": "choices 1"}, {"content": "choices 2"}]

    with pytest.raises(ValueError):
        model._chat_completions(str)
