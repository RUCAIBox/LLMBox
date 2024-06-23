from logging import getLogger
from typing import Optional, Type

import qianfan
from qianfan.resources.typing import QfResponse

from ..utils import ModelArguments
from .model import ApiModel, RaiseError, RetryError, SkipError, ensure_type

logger = getLogger(__name__)


class Qianfan(ApiModel):
    r"""The model for calling Qianfan APIs. (Baidu)

    Please refer to https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html.

    We now support `ERNIE-4.0-8K`, `ERNIE-3.5-8K`, `ERNIE-3.5-8K-0205`, `ERNIE-3.5-8K-1222`,
                   `ERNIE-3.5-4K-0205`, `ERNIE-Speed-8K`, `ERNIE-Speed-128K`, `ERNIE-Lite-8K-0922`,
                   `ERNIE-Lite-8K-0308`, `ERNIE Tiny`, `ERNIE Speed-AppBuilder`.
    """

    model_backend = "qianfan"
    model: qianfan.ChatCompletion

    _raise_errors = (RaiseError,)
    _retry_errors = (RetryError,)
    _skip_errors = (SkipError,)

    _repr = ["model_type", "model_backend", "multi_turn"]

    def __init__(self, args: ModelArguments):
        super().__init__(args)

        logger.info(f"Trying to load Qianfan model")
        self.model = qianfan.ChatCompletion(access_key=args.qianfan_access_key, secret_key=args.qianfan_secret_key)

    @staticmethod
    def _get_error_type(response: QfResponse) -> Optional[Type[Exception]]:
        # https://cloud.baidu.com/doc/WENXINWORKSHOP/s/tlmyncueh
        if "error_code" not in response.body:
            return None
        if (
            response.body["error_code"] in {336007, 336103, 336122}
            or "inappropriate content" in response.body["message"]
        ):
            return SkipError
        elif response.body["error_code"] in {1, 2, 4, 17, 18, 19, 336000, 336100, 336120}:
            return RetryError
        else:
            return RaiseError

    @staticmethod
    @ensure_type(str)
    def _get_assistant(msg: QfResponse) -> str:
        return msg.body["result"]

    def _chat_completions(self, *, messages, model, **kwargs) -> QfResponse:
        return self.model.do(messages=messages, model=model, **kwargs)
