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
        self.type = "instruction"
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_try_times = 5

    def set_generation_args(self, **extra_model_args):
        """Set the configurations for open-ended generation. This is useful because different datasets may have different requirements for generation."""
        generation_kwargs = {}
        for key in [
            "temperature",
            "top_p",
            "top_k",
            "penalty_score",
            "stop",
            "disable_search",
            "enable_citation",
            "max_tokens"
        ]:
            # ModelArguments > extra_model_args
            value = getattr(self.args, key, None)
            if value is None:
                value = extra_model_args.pop(key, None)

            if key == "max_tokens":
                key = "max_output_tokens"
                value = 1024 if value is None else value
            if key == "temperature":
                value = max(0.0001, value)
            if value is not None:
                generation_kwargs[key] = value

        self.generation_kwargs = generation_kwargs

    def generation(self, batched_inputs):
        results = self.request(batched_inputs, self.generation_kwargs)
        answers = []
        for result in results:
            answer = result[0]["result"]
            answers.append(answer)
        return answers

    def request(self, prompt, kwargs):
        r"""Call the Qianfan API.

        Args:
            prompt (List[str]): The list of input prompts.
            model_args (dict): The additional calling configurations.

        Returns:
            List[dict]: The responsed JSON results.
        """
        qianfan.AccessKey(self.qianfan_access_key)
        qianfan.SecretKey(self.qianfan_secret_key)
        chat_comp = qianfan.ChatCompletion()
        for _ in range(self.max_try_times):
            message = [{"role": "user", "content": prompt[0]}]
            response = chat_comp.do(
                model=self.name,
                messages=message,
                **kwargs
            )
            if "error_code" in response:
                logger.warning(response.error_msg)
                logger.warning("retrying...")
                time.sleep(1)
            else:
                return [[response.body]]
        raise ConnectionError("Qianfan API error")
