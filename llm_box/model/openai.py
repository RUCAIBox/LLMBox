import time
from logging import getLogger

import openai
import tiktoken

from ..utils import ModelArguments
from .model import Model

logger = getLogger(__name__)


class Openai(Model):
    r"""The model for calling OpenAI APIs.

    Please refer to https://platform.openai.com/docs/models.

    We now support base GPT-3 models (`ada`, `babbage`, `curie', `davinci`, `babbage-002`, and `davinci-002`).
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
        self.name = args.model_name_or_path
        self.type = "instruction" if self.name in [
            "gpt-3.5-turbo", "gpt-3.5-turbo-instruct", "text-davinci-003"
        ] else "base"
        self.tokenizer = tiktoken.get_encoding(tiktoken.encoding_name_for_model(self.name))
        # TODO: compatible for gpt-3.5-turbo (enum_type?)
        self.max_tokens = args.max_new_tokens or 2048
        self.max_try_times = 5
        self.temperature = args.temperature

    def set_ppl_args(self, **kwargs):
        r"""Set the configurations for PPL score calculation. This is useful because different datasets may have different requirements for ppl calculation."""
        # TODO: gpt-3.5-turbo doesn't support echo and logprobs, and it doesn't support max_tokens=0
        self.ppl_kwargs = dict(echo=True, max_tokens=0, logprobs=0)

    def set_generation_args(self, **kwargs):
        r"""Set the configurations for open-ended generation. This is useful because different datasets may have different requirements for generation."""
        self.generation_kwargs = dict(max_tokens=self.max_tokens, temperature=self.temperature)

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
                # TODO: compatible for gpt-3.5-turbo
                if self.name == "gpt-3.5-turbo":
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
            except (Exception, KeyboardInterrupt) as e:
                logger.warning(f'Receive {e.__class__.__name__}, retrying...')
                time.sleep(1)
        raise ConnectionError("OpenAI API error")

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
        prompt = [question for question in batch]
        results = self.request(prompt, self.generation_kwargs)
        answers = []
        for result, _ in zip(results, batch):
            if self.name == 'gpt-3.5-turbo':
                answer = result[0]['message']['content']
            else:
                answer = result['text']
            answers.append(answer)
        return answers
