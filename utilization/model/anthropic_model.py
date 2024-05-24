import os
from logging import getLogger

import anthropic
from anthropic.types import Message

from ..utils import ModelArguments
from .model import ApiModel

logger = getLogger(__name__)


class Anthropic(ApiModel):
    r"""The model for calling Anthropic APIs.

    Please refer to https://docs.anthropic.com/claude/reference.

    We now support Claude (`claude-2.1`) and Claude Instant (`claude-instant-1.2`).
    """

    model_backend = "anthropic"
    model: anthropic.Anthropic

    _retry_errors = (anthropic.APITimeoutError, anthropic.RateLimitError)
    _raise_errors = (
        anthropic.APIConnectionError, anthropic.AuthenticationError, anthropic.BadRequestError, anthropic.ConflictError,
        anthropic.NotFoundError, anthropic.PermissionDeniedError, anthropic.UnprocessableEntityError
    )
    _skip_errors = (anthropic.InternalServerError,)

    _repr = ["model_type", "model_backend", "multi_turn"]

    def __init__(self, args: ModelArguments):
        super().__init__(args)

        base_url = os.getenv("ANTHROPIC_BASE_URL", None)
        logger.info(f"Trying to load Anthropic model with ANTHROPIC_BASE_URL='{base_url}'")

        self.model = anthropic.Anthropic(api_key=args.anthropic_api_key, base_url=base_url)

    def _chat_completions(self, *, messages, model, **kwargs):
        return self.model.messages.create(messages=messages, model=model, **kwargs)

    @staticmethod
    def _get_assistant(msg: Message) -> str:
        return msg.content[0].text
