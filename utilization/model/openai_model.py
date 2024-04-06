import time
from copy import copy
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

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

    tokenizer: tiktoken.Encoding

    def __init__(self, args: ModelArguments):
        super().__init__(args)

        if openai.__version__ != "0.28.1":
            logger.warning(
                f"OpenAI version is {openai.__version__}, not 0.28.1. Please make sure the version is correct."
            )

        logger.info(f"Trying to load OpenAI model with api_base='{openai.api_base}'")
        self.api_key = openai.api_key  # the actual api key is used in icl

        self.args = args
        self.name = args.model_name_or_path
        self.type = "instruction" if self.name in OPENAI_INSTRUCTION_MODELS else "base"
        self.is_chat_model = self.name in OPENAI_CHAT_MODELS
        self.tokenizer = tiktoken.get_encoding(args.tokenizer_name_or_path)
        self.max_try_times = 5

    def set_ppl_args(self, **extra_model_args):
        r"""Set the configurations for PPL score calculation."""
        # TODO: GPT-3.5 series models don't support echo and logprobs
        if "gpt-3.5" in self.name:
            raise ValueError(
                f"{self.name} doesn't support PPL score calculation. Please use get_prob mode by setting `--ranking_type prob` instead"
            )
        self.ppl_kwargs = dict(echo=True, max_tokens=0, logprobs=0)
        self.multi_turn = extra_model_args.pop("multi_turn", False)

    def set_generation_args(self, **extra_model_args):
        r"""Set the configurations for open-ended generation. This is useful because different datasets may have different requirements for generation."""
        generation_kwargs = {}
        for key in [
            "temperature",
            "top_p",
            "max_tokens",
            "best_of",
            "frequency_penalty",
            "presence_penalty",
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
        self.multi_turn = extra_model_args.pop("multi_turn", False)

    def set_prob_args(self, **extra_model_args):

        self._word_label_ids = []
        self._token_label_ids = []
        self._word_label_texts = []
        self._token_label_texts = []
        self._candidate_ids = extra_model_args.pop("candidate_ids", None)
        self._candidate_texts = self.tokenizer.decode_batch(self._candidate_ids) if self._candidate_ids else None

        self.prob_kwargs = {"max_tokens": 1, "temperature": 0.0}
        self.constant_option_num = extra_model_args.pop("constant_option_num", False)

        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")

    def _get_label_ids(self, option_num: Optional[int]) -> Tuple[List[int], List[str]]:
        """Return the tokenized labels of options and labels themselves."""
        if option_num is not None:
            if len(self._word_label_ids) < option_num:
                labels = []
                for i in range(len(self._word_label_ids), option_num):
                    word = chr(i + 65)
                    token = " " + chr(i + 65)
                    self._word_label_texts.append(word)
                    self._token_label_texts.append(token)
                    labels.append(word + token)
                word_labels, token_labels = zip(*self.tokenizer.encode_batch(labels))
                self._word_label_ids.extend(word_labels)
                self._token_label_ids.extend(token_labels)

            ids = self._word_label_ids[:option_num] + self._token_label_ids[:option_num]
            texts = self._word_label_texts[:option_num] + self._token_label_texts[:option_num]
            return ids, texts
        else:
            if self._candidate_ids is None:
                raise ValueError("The candidate_ids must be provided when option_num is None.")
            return self._candidate_ids, self._candidate_texts

    def get_ppl(self, batched_inputs: List[Tuple[str, ...]]) -> List[Tuple[float, int]]:
        prompt = ["".join(parts) for parts in batched_inputs]

        results = self.request(prompt, self.ppl_kwargs)

        ppls = []
        for result, (src, _) in zip(results, batched_inputs):
            tgt_start = max(1, result["logprobs"]["text_offset"].index(len(src)))  # designed for src=''
            tgt_end = len(result["logprobs"]["text_offset"])
            ppl = -sum(result["logprobs"]["token_logprobs"][tgt_start:])
            ppls.append((ppl, tgt_end - tgt_start))
        return ppls

    def generation(self, batched_inputs: List[str]) -> Union[List[Tuple[str]], List[str]]:
        results = self.request(batched_inputs, self.generation_kwargs, multi_turn=self.multi_turn)
        answers = []
        for result in results:
            if self.is_chat_model:
                answer = result[0]["message"]["content"]
            else:
                answer = result["text"]
            answers.append(answer)
        return [tuple(answers)] if self.multi_turn else answers

    def get_prob(self, batched_inputs: List[Tuple[str, int]]) -> List[List[int]]:

        *batched_prompts, batched_option_nums = map(list, zip(*batched_inputs))
        batch_size = len(batched_prompts[0])
        batched_prompts = ["".join(group[idx] for group in batched_prompts) for idx in range(batch_size)]
        if self.constant_option_num:
            label_ids, label_texts = self._get_label_ids(batched_option_nums[0])
            label_texts = [label_texts] * len(batched_option_nums)
            logit_bias = dict.fromkeys(label_ids, 100)
        else:
            labels = [self._get_label_ids(b) for b in batched_option_nums]
            label_texts = [l[1] for l in labels]
            logit_bias = [dict.fromkeys(l[0], 100) for l in labels]

        kwargs = copy(self.prob_kwargs)
        if self.is_chat_model:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = len(label_texts)
        else:
            kwargs["logprobs"] = len(label_texts)

        results = self.request(batched_prompts, self.prob_kwargs, logit_bias=logit_bias)

        answers = []
        for result, option_num, label in zip(results, batched_option_nums, label_texts):
            result = result[0] if self.is_chat_model else result
            if result["logprobs"] is not None:

                probs = [-9999.] * (option_num * 2)
                if self.is_chat_model:
                    top_logprobs = result["logprobs"]["content"][0]["top_logprobs"]
                    for token in top_logprobs:
                        if token["token"] in label:
                            probs[label.index(token["token"])] = token["logprob"]
                else:
                    top_logprobs = result["logprobs"]["top_logprobs"][0]
                    for l, p in top_logprobs.items():
                        if l in label:
                            probs[label.index(l)] = p

            else:
                probs = [-9999.] * (option_num * 2)
                if self.is_chat_model:
                    text = result["message"]["content"]
                else:
                    text = result["text"]
                probs[label.index(text)] = 20.0

            answers.append(probs)
        return answers

    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        logit_bias: Optional[Dict[int, float]] = None,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        **kwargs
    ) -> List[dict]:
        # reference: https://platform.openai.com/docs/api-reference/chat/create
        return openai.ChatCompletion.create(
            model=self.name,
            messages=messages,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            **kwargs,
        )["choices"]

    def _completion(
        self,
        prompt: List[str],
        *,
        logit_bias: Optional[Dict[int, float]] = None,
        logprobs: Optional[int] = None,
        **kwargs
    ) -> List[dict]:
        # reference: https://platform.openai.com/docs/api-reference/completions/create
        return openai.Completion.create(
            model=self.name,
            prompt=prompt,
            logit_bias=logit_bias,
            logprobs=logprobs,
            **kwargs,
        )["choices"]

    def request(
        self,
        prompt: List[str],
        openai_kwargs: Dict[str, Any],
        *,
        multi_turn: bool = False,
        logit_bias: Union[Dict[int, float], List[Dict[int, float]], None] = None,
    ) -> Union[List[dict], List[List[dict]]]:
        r"""Call the OpenAI API.

        Args:
            prompt (List[str]): The list of input prompts.
            openai_kwargs (dict): The additional calling configurations.
            multi_turn (bool): Default is False. Set to True if multi-turns needed.

        Returns:
            List[dict]: The list of results.
        """
        same_logit_bias = not isinstance(logit_bias, list)
        if not same_logit_bias:
            assert len(prompt) == len(logit_bias), "The length of prompt and logit_bias should be the same."

        for _ in range(self.max_try_times):
            try:
                # openai chat-based model does not support batch size > 1
                if self.is_chat_model:
                    messages = []
                    results = []
                    parts = prompt[0].split("__SEPARATOR__") if multi_turn else prompt
                    for idx, query in enumerate(parts):
                        if len(query) == 0:
                            continue
                        messages.append({"role": "user", "content": query})
                        lb = logit_bias if same_logit_bias else logit_bias[idx]

                        msg = self._chat_completion(
                            messages=messages,
                            logit_bias=lb,
                            **openai_kwargs,
                        )

                        results.append(msg)
                        messages.append({"role": "assistant", "content": msg[0]["message"]["content"]})
                    return results
                else:
                    if same_logit_bias:
                        lb = logit_bias if same_logit_bias else logit_bias[idx]
                        results = self._completion(prompt, logit_bias=logit_bias, **openai_kwargs)
                    else:
                        results = [
                            self._completion(p, logit_bias=lb, **openai_kwargs)[0] for p, lb in zip(prompt, logit_bias)
                        ]
                    return results
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
