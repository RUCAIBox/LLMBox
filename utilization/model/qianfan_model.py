import time
from logging import getLogger

import qianfan
import tiktoken

from ..utils import ModelArguments
from .model import Model

logger = getLogger(__name__)


class Qianfan(Model):
    r"""The model for calling Qianfan APIs. (Baidu)

    Please refer to https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html.

    We now support `ERNIE-4.0-8K`, `ERNIE-3.5-8K`, `ERNIE-3.5-8K-0205`, `ERNIE-3.5-8K-1222`,
                   `ERNIE-3.5-4K-0205`, `ERNIE-Speed-8K`, `ERNIE-Speed-128K`, `ERNIE-Lite-8K-0922`,
                   `ERNIE-Lite-8K-0308`, `ERNIE Tiny`, `ERNIE Speed-AppBuilder`.
    """

    model_backend = "qianfan"

    _repr = ["model_type", "model_backend", "multi_turn"]

    def __init__(self, args: ModelArguments):
        super().__init__(args)
        if not args.qianfan_access_key or not args.qianfan_secret_key:
            raise ValueError(
                "Qianfan API access key and secret key is required. Please set it by passing `--qianfan_access_key` and `--qianfan_secret_key` or through environment variable `QIANFAN_ACCESS_KEY` and `QIANFAN_SECRET_KEY`."
            )
        private_key = args.qianfan_access_key[:6] + "*" * 14 + args.qianfan_access_key[-4:]
        logger.info(f"Trying to load Qianfan model with access_key='{private_key}' and secret_key " + "*" * 32)
        self.qianfan_access_key = args.qianfan_access_key
        self.qianfan_secret_key = args.qianfan_secret_key

        self.args = args
        self.name = args.model_name_or_path
        self.tokenizer = tiktoken.get_encoding(args.tokenizer_name_or_path)
        self.max_try_times = 10

    def set_generation_args(self, **extra_model_args):
        """Set the configurations for open-ended generation. This is useful because different datasets may have different requirements for generation."""
        generation_kwargs = {}
        for key in [
            "temperature", "top_p", "top_k", "penalty_score", "stop", "disable_search", "enable_citation", "max_tokens"
        ]:
            # ModelArguments > extra_model_args
            value = getattr(self.args, key, None)
            if value is None:
                value = extra_model_args.pop(key, None)

            if key == "max_tokens":
                key = "max_output_tokens"
                value = 1024 if value is None else value
            if value is not None:
                generation_kwargs[key] = value

        if "temperture" not in generation_kwargs:
            generation_kwargs["temperature"] = 0.0001
        self.generation_kwargs = generation_kwargs
        self.multi_turn = extra_model_args.pop("multi_turn", False)
        return self.generation_kwargs

    def generation(self, batched_inputs):
        results = self.request(batched_inputs, self.generation_kwargs, multi_turn=self.multi_turn)
        answers = []
        for result in results:
            answer = result["result"]
            answers.append(answer)
        return [tuple(answers)] if self.multi_turn else answers

    def request(self, prompt, kwargs, multi_turn=False):
        r"""Call the Qianfan API.

        Args:
            prompt (List[str]): The list of input prompts.
            model_args (dict): The additional calling configurations.
            multi_turn (bool): Default is False. Set to True if multi-turns needed.

        Returns:
            List[dict]: The responsed JSON results.
        """
        qianfan.AccessKey(self.qianfan_access_key)
        qianfan.SecretKey(self.qianfan_secret_key)
        chat_comp = qianfan.ChatCompletion()
        for _ in range(self.max_try_times):
            error_msg = "EMPTY_ERROR_MSG"
            try:
                messages = []
                results = []
                parts = prompt[0].split("__SEPARATOR__") if multi_turn else prompt
                for query in parts:
                    if len(query) == 0:
                        continue
                    messages.append({"role": "user", "content": query})
                    msg = chat_comp.do(model=self.name, messages=messages, **kwargs)
                    if "error_code" in msg:
                        error_msg = msg.error_msg
                    assert ("error_code" not in msg)
                    results.append(msg.body)
                    messages.append({"role": "assistant", "content": msg.body["result"]})
                return results
            except Exception as e:
                logger.warning("Receive error: {}".format(error_msg))
                logger.warning("retrying...")
                time.sleep(1)
        raise ConnectionError("Qianfan API error")
