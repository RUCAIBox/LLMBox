import time
from logging import getLogger

import openai
import tiktoken

from ..utils import ModelArguments
from .enum import OPENAI_CHAT_MODELS, OPENAI_INSTRUCTION_MODELS
from .model import Model

logger = getLogger(__name__)


class Openai(Model):
    r"""The model for calling OpenAI APIs.

    Please refer to https://platform.openai.com/docs/models.

    We now support GPT-3 (`babbage-002` and `davinci-002`) and GPT-3.5 series models (`gpt-3.5-turbo-instruct`, `gpt-3.5-turbo`, `gpt-3.5-turbo-1106`, and `gpt-3.5-turbo-16k`).
    """

    def __init__(self, args: ModelArguments):
        super().__init__(args)
        if not args.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Please set it by passing a `--openai_api_key` or through environment variable `OPENAI_API_KEY`."
            )
        logger.info(f"Trying to load OpenAI model with api_key='{args.openai_api_key}' and base='{openai.api_base}'")
        self.api_key = openai.api_key  # the actual api key is used in icl

        self.args = args
        self.name = args.model_name_or_path
        self.type = "instruction" if self.name in OPENAI_INSTRUCTION_MODELS else "base"
        self.tokenizer = tiktoken.get_encoding(tiktoken.encoding_name_for_model(self.name))
        self.max_try_times = 5

    def set_ppl_args(self, **kwargs):
        r"""Set the configurations for PPL score calculation."""
        # TODO: GPT-3.5 series models don't support echo and logprobs
        self.ppl_kwargs = dict(echo=True, max_tokens=0, logprobs=0)

    def set_generation_args(self, **kwargs):
        r"""Set the configurations for open-ended generation. This is useful because different datasets may have different requirements for generation."""
        generation_kwargs = {}
        for key in ["temperature", "top_p", "max_tokens", "best_of", "frequency_penalty", "presence_penalty", "stop"]:
            value = getattr(self.args, key) if getattr(self.args, key, None) is not None else kwargs.get(key, None)
            if key == "max_tokens" and value is None:
                value = 1024
            if value is not None:
                generation_kwargs[key] = value
        if generation_kwargs.get("temperature", 1) == 0:
            generation_kwargs["seed"] = self.args.seed
        self.generation_kwargs = generation_kwargs

    def get_ppl(self, batched_inputs):
        prompt = [src + tgt for src, tgt in batched_inputs]
        results = self.request(prompt, self.ppl_kwargs)
        ppls = []
        for result, (src, _) in zip(results, batched_inputs):
            tgt_start = max(1, result["logprobs"]["text_offset"].index(len(src)))  # designed for src=''
            tgt_end = len(result["logprobs"]["text_offset"])
            ppl = -sum(result["logprobs"]["token_logprobs"][tgt_start:])
            ppls.append((ppl, tgt_end - tgt_start))
        return ppls

    def generation(self, batched_inputs):
        results = self.request(batched_inputs, self.generation_kwargs)
        answers = []
        for result in results:
            if self.name in OPENAI_CHAT_MODELS:
                answer = result[0]["message"]["content"]
            else:
                answer = result["text"]
            answers.append(answer)
        return answers

    def request(self, prompt, openai_kwargs):
        r"""Call the OpenAI API.

        Args:
            prompt (List[str]): The list of input prompts.
            openai_kwargs (dict): The additional calling configurations.

        Returns:
            List[dict]: The responsed JSON results.
        """
        for _ in range(self.max_try_times):
            try:
                if self.name in OPENAI_CHAT_MODELS:
                    message = [{"role": "user", "content": prompt[0]}]
                    response = openai.ChatCompletion.create(model=self.name, messages=message, **openai_kwargs)
                    return [response["choices"]]
                else:
                    response = openai.Completion.create(model=self.name, prompt=prompt, **openai_kwargs)
                    return response["choices"]
            except openai.error.RateLimitError:
                logger.warning("Receive openai.error.RateLimitError, retrying...")
                time.sleep(10)
            except openai.error.AuthenticationError as e:
                raise e
            except openai.error.InvalidRequestError as e:
                raise e
            except Exception as e:
                logger.warning(f"Receive {e.__class__.__name__}: {str(e)}")
                logger.warning("retrying...")
                time.sleep(1)
        raise ConnectionError("OpenAI API error")
