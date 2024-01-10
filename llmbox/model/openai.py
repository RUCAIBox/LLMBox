import time
import openai
import tiktoken
from logging import getLogger

from ..utils import ModelArguments
from .model import Model
from .enum import OPENAI_INSTRUCTION_MODELS, OPENAI_CHAT_MODELS

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
        openai.api_key = args.openai_api_key
        secret_key = openai.api_key[:8] + "*" * 39 + openai.api_key[-4:]
        logger.info(f"OpenAI API key: {secret_key}, base: {openai.api_base}")
        self.api_key = openai.api_key

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
        for key in ['temperature', 'top_p', 'max_tokens', 'best_of', 'frequency_penalty', 'presence_penalty', 'seed']:
            value = getattr(self.args, key, None) or kwargs.get(key, None)
            if value:
                generation_kwargs[key] = value
        self.generation_kwargs = generation_kwargs

    def get_ppl(self, batch):
        prompt = [src + tgt for src, tgt in batch]
        results = self.request(prompt, self.ppl_kwargs)
        ppls = []
        for result, (src, _) in zip(results, batch):
            tgt_start = result['logprobs']['text_offset'].index(len(src))
            tgt_end = len(result['logprobs']['text_offset'])
            ppl = -sum(result['logprobs']['token_logprobs'][tgt_start:])
            ppls.append((ppl, tgt_end - tgt_start))
        return ppls

    def generation(self, batch):
        results = self.request(batch, self.generation_kwargs)
        answers = []
        for result in results:
            if self.name in OPENAI_CHAT_MODELS:
                answer = result[0]['message']['content']
            else:
                answer = result['text']
            answers.append(answer)
        return answers

    def request(self, prompt, model_args):
        r"""Call the OpenAI API.

        Args:
            prompt (List[str]): The list of input prompts.
            model_args (dict): The additional calling configurations.

        Returns:
            List[dict]: The responsed JSON results.
        """
        for _ in range(self.max_try_times):
            try:
                if self.name in OPENAI_CHAT_MODELS:
                    message = [{'role': 'user', 'content': prompt[0]}]
                    response = openai.ChatCompletion.create(model=self.name, messages=message, **model_args)
                    return [response["choices"]]
                else:
                    response = openai.Completion.create(model=self.name, prompt=prompt, **model_args)
                    return response["choices"]
            except openai.error.RateLimitError:
                logger.warning('Receive openai.error.RateLimitError, retrying...')
                time.sleep(10)
            except openai.error.AuthenticationError as e:
                raise e
            except openai.error.InvalidRequestError as e:
                raise e
            except Exception as e:
                logger.warning(f'Receive {e.__class__.__name__}, retrying...')
                time.sleep(1)
        raise ConnectionError("OpenAI API error")
