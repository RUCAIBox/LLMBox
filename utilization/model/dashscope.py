import time
from logging import getLogger
from http import HTTPStatus

import dashscope

from ..utils import ModelArguments
from .model import Model

logger = getLogger(__name__)


class Dashscope(Model):
    r"""The model for calling Dashscope APIs. (Aliyun)

    Please refer to https://help.aliyun.com/zh/dashscope/.

    We now support `qwen-turbo`, `qwen-plus`, `qwen-max`, `qwen-max-1201`, `qwen-max-longcontext`, `qwen1.5-72b-chat`,
                   `qwen1.5-14b-chat`, `qwen1.5-7b-chat`, `qwen-72b-chat`, `qwen-14b-chat`, `qwen-7b-chat`,
                   `qwen-1.8b-longcontext-chat`, `qwen-1.8b-chat`.
    """

    def __init__(self, args: ModelArguments):
        super().__init__(args)
        if not args.dashscope_api_key:
            raise ValueError(
                "Dashscope API key is required. Please set it by passing a `--dashscope_api_key` or through environment variable `DASHSCOPE_API_KEY`."
            )
        private_key = args.dashscope_api_key[:8] + "*" * 23 + args.dashscope_api_key[-4:]
        logger.info(f"Trying to load Dashscope model with api_key='{private_key}'")
        self.api_key = args.dashscope_api_key

        self.args = args
        self.name = args.model_name_or_path
        self.type = "instruction"
        self.tokenizer = dashscope.get_tokenizer(self.name)
        self.max_try_times = 10

    def set_generation_args(self, **extra_model_args):
        """Set the configurations for open-ended generation. This is useful because different datasets may have different requirements for generation."""
        generation_kwargs = {}
        for key in [
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "repetition_penalty",
            "enable_search",
            "stop",
        ]:
            # ModelArguments > extra_model_args
            value = getattr(self.args, key, None)
            if value is None:
                value = extra_model_args.pop(key, None)

            if key == "max_tokens" and value is None:
                value = 1024
            if value is not None:
                generation_kwargs[key] = value

        if generation_kwargs.get("temperature", 1) == 0:
            generation_kwargs["seed"] = self.args.seed
        self.generation_kwargs = generation_kwargs

    def generation(self, batched_inputs):
        results = self.request(batched_inputs, self.generation_kwargs)
        answers = []
        for result in results:
            answer = result[0].content
            answers.append(answer)
        return answers

    def request(self, prompt, kwargs):
        r"""Call the DashScope API.

        Args:
            prompt (List[str]): The list of input prompts.
            model_args (dict): The additional calling configurations.

        Returns:
            List[dict]: The responsed JSON results.
        """
        for _ in range(self.max_try_times):
            message = [{"role": "user", "content": prompt[0]}]
            response = dashscope.Generation.call(
                model=self.name,
                messages=message,
                result_format="message",
                **kwargs
            )
            if response.status_code == HTTPStatus.OK:
                return [[response.output.choices[0].message]]
            else:
                logger.warning(response.message)
                logger.warning("retrying...")
                time.sleep(1)
        raise ConnectionError("Dashscope API error")
