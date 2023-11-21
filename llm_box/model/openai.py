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
        self.name = args.model_name_or_path
        self.type = "base"
        self.tokenizer = tiktoken.get_encoding(tiktoken.encoding_name_for_model(self.name))
        # TODO: compatible for gpt-3.5-turbo (enum_type?)
        self.max_tokens = 2048
        self.max_try_times = 5
        self.temperature = 0.7 if args.use_sc else 1.0
        # TODO: gpt-3.5-turbo doesn't support echo and logprobs, and it doesn't support max_tokens=0
        self.ppl_kwargs = dict(echo=True, max_tokens=0, logprobs=0)

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
            except Exception as e:
                print(e)
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
