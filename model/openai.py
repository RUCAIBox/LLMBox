import os
import time
import openai
import tiktoken
from .model import Model


class Openai(Model):
    r"""The model for calling OpenAI APIs.

    Please refer to https://platform.openai.com/docs/models.

    We now support base GPT-3 models (`ada`, `babbage`, `curie', `davinci`, `babbage-002`, and `davinci-002`).
    """

    def __init__(self, args):
        super().__init__(args)
        openai.api_key = os.environ.get("OPENAI_API_SECRET_KEY") or args.openai_api_key
        self.name = args.model
        self.type = "base"
        # TODO: compatible for gpt-3.5-turbo
        self.tokenizer = tiktoken.get_encoding("r50k_base")
        # TODO: compatible for gpt-3.5-turbo (enum_type?)
        self.max_tokens = 2048
        self.max_try_times = 5

        ppl_default_kwargs = dict(echo=True, max_tokens=0, logprobs=0)
        self.ppl_kwargs = {**ppl_default_kwargs, **self.args.kwargs}
        self.generation_kwargs = self.args.kwargs

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
                response = openai.Completion.create(model=self.name, prompt=prompt, **model_args)
                return response["choices"]
            except openai.error.RateLimitError:
                print('openai.error.ServiceUnavailableError\nRetrying...')
                time.sleep(10)
            except openai.error.ServiceUnavailableError:
                print('openai.error.ServiceUnavailableError\nRetrying...')
                time.sleep(1)
            except openai.error.Timeout:
                print('openai.error.Timeout\nRetrying...')
                time.sleep(1)
            except openai.error.APIError:
                print('openai.error.APIError\nRetrying...')
                time.sleep(1)
            except openai.error.APIConnectionError:
                print('openai.error.APIConnectionError\nRetrying...')
                time.sleep(1)
            except:
                print("UnknownError")
                time.sleep(1)
        raise ConnectionError("OpenAI API error")

    def get_ppl(self, batch):
        prompt = [src + tgt for src, tgt in batch]
        results = self.request(prompt, self.ppl_kwargs)
        ppls = []
        for result, (src, _) in zip(results, batch):
            tgt_start = result['logprobs']['text_offset'].index(len(src))
            tgt_end = len(result['logprobs']['text_offset'])
            ppl = -sum(result['logprobs']['token_logprobs'][tgt_start:]) / (tgt_end - tgt_start)
            ppls.append(ppl)
        return ppls

    def generation(self, batch):
        pass
