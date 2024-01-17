import time
from logging import getLogger

import anthropic
import tiktoken

from ..utils import ModelArguments
from .model import Model

logger = getLogger(__name__)


class Anthropic(Model):
    r"""The model for calling Anthropic APIs.
    
    Please refer to https://docs.anthropic.com/claude/reference.
    
    We now support Claude (`claude-2.1`) and Claude Instant (`claude-instant-1.2`).
    """

    def __init__(self, args: ModelArguments):
        super().__init__(args)
        if not args.anthropic_api_key:
            raise ValueError(
                "Anthropic API key is required. Please set it by passing a `--anthropic_api_key` or through environment variable `ANTHROPIC_API_KEY`."
            )
        self.api_key = args.anthropic_api_key

        self.args = args
        self.name = args.model_name_or_path
        self.type = "base"
        self.tokenizer = tiktoken.get_encoding(tiktoken.encoding_name_for_model(self.name))
        self.max_try_times = 5
    
    def request(self, prompt, model_args):
        r"""Call the Anthropic API.

        Args:
            prompt (List[str]): The list of input prompts.
            model_args (dict): The additional calling configurations.

        Returns:
            List[dict]: The responsed JSON results.
        """
        client = anthropic.Anthropic(api_key=self.api_key)
        for _ in range(self.max_try_times):
            try:
                message = [{"role": "user", "content": prompt[0]}]
                response = client.beta.messages.create(model=self.name, messages=message, **model_args)
                return [response.content]
            except anthropic.RateLimitError:
                logger.warning('Receive anthropic.RateLimitError, retrying...')
                time.sleep(10)
            except anthropic.APIStatusError as e:
                logger.warning("Another non-200-range status code was received")
                raise(e)
            except anthropic.APIConnectionError as e:
                logger.warning("The server could not be reached")
                raise(e)
            except Exception as e:
                logger.warning(f'Receive {e.__class__.__name__}: {str(e)}')
                logger.warning('retrying...')
                time.sleep(1)
        raise ConnectionError("Anthropic API error")