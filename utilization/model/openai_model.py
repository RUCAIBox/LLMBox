import os
import re
from copy import copy
from logging import getLogger
from typing import Any, List, Optional, Tuple, Union

import openai
from openai.types import CompletionChoice
from openai.types.chat.chat_completion import Choice

from ..utils import ModelArguments
from .model import ApiModel
from .model_enum import OPENAI_CHAT_COMPLETIONS_ARGS, OPENAI_COMPLETIONS_ARGS

logger = getLogger(__name__)


class Openai(ApiModel):
    r"""The model for calling OpenAI APIs.

    Please refer to https://platform.openai.com/docs/models.

    We now support GPT-3 (`babbage-002` and `davinci-002`) and GPT-3.5 series models (`gpt-3.5-turbo-instruct`, `gpt-3.5-turbo`, `gpt-3.5-turbo-1106`, and `gpt-3.5-turbo-16k`).
    """

    model_backend = "openai"

    _retry_errors = (openai.APITimeoutError, openai.InternalServerError, openai.RateLimitError)
    _raise_errors = (
        openai.APIConnectionError, openai.AuthenticationError, openai.BadRequestError, openai.ConflictError,
        openai.NotFoundError, openai.PermissionDeniedError, openai.UnprocessableEntityError
    )
    _skip_errors = ()

    _repr = ["model_type", "model_backend", "multi_turn", "candidate_ids"]

    def __init__(self, args: ModelArguments):
        super().__init__(args)

        base_url = os.getenv("OPENAI_BASE_URL", None)
        logger.info(f"Trying to load OpenAI model with OPENAI_BASE_URL='{base_url}'")

        self.model = openai.OpenAI(api_key=openai.api_key, base_url=base_url)

    def _chat_completions(self, *, messages, model, **kwargs):
        results = openai.chat.completions.create(messages=messages, model=model, **kwargs)
        if hasattr(results, "choices"):
            return results.choices
        else:
            raise ValueError(f"Unexpected response from OpenAI API: {results}")

    def _completions(self, **kwargs):
        results = openai.completions.create(**kwargs)
        if hasattr(results, "choices"):
            return results.choices
        else:
            raise ValueError(f"Unexpected response from OpenAI API: {results}")

    @staticmethod
    def _get_assistant(msg: Union[List[Choice], CompletionChoice]) -> str:
        if isinstance(msg, CompletionChoice):
            return msg.text
        else:
            return msg[0].message.content

    def set_ppl_args(self, **extra_model_args):
        r"""Set the configurations for PPL score calculation."""

        self.ppl_kwargs = dict(echo=True, max_tokens=0, logprobs=0)

        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")
        return self.ppl_kwargs

    def set_prob_args(self, **extra_model_args):

        self._word_label_ids = []
        self._token_label_ids = []
        self._word_label_texts = []
        self._token_label_texts = []
        self._option_regex = []
        self.candidate_ids = extra_model_args.pop("candidate_ids", None)
        self._candidate_texts = self.tokenizer.decode_batch(self.candidate_ids) if self.candidate_ids else None

        self.prob_kwargs = {"max_tokens": 1, "temperature": 0.0}
        self.constant_option_num = extra_model_args.pop("constant_option_num", False)

        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")
        return self.prob_kwargs

    def _get_label_ids(self, option_num: Optional[int]) -> Tuple[List[int], List[str]]:
        """Return the tokenized labels of options and labels themselves."""
        matches = r"\b([A-{op}])\b|\b([A-{op}])[\u2E80-\u9FFF]|[\u2E80-\u9FFF]([A-{op}])\b|[\u2E80-\u9FFF]([A-{op}])[\u2E80-\u9FFF]"
        if option_num is not None:
            if len(self._word_label_ids) < option_num:
                labels = []
                regexs = []
                for i in range(len(self._word_label_ids), option_num):
                    word = chr(i + 65)
                    token = " " + chr(i + 65)
                    self._word_label_texts.append(word)
                    self._token_label_texts.append(token)
                    regexs.append(re.compile(matches.format(op=chr(ord("A") + i))))
                    labels.append(word + token)
                word_labels, token_labels = zip(*self.tokenizer.encode_batch(labels))
                self._word_label_ids.extend(word_labels)
                self._token_label_ids.extend(token_labels)
                self._option_regex.extend(regexs)

            ids = self._word_label_ids[:option_num] + self._token_label_ids[:option_num]
            texts = self._word_label_texts[:option_num] + self._token_label_texts[:option_num]
            return ids, texts
        else:
            if self.candidate_ids is None:
                raise ValueError("The candidate_ids must be provided when option_num is None.")
            return self.candidate_ids, self._candidate_texts

    def get_ppl(self, batched_inputs: List[Tuple[str, ...]]) -> List[Tuple[float, int]]:
        prompt = ["".join(parts) for parts in batched_inputs]

        results: List[CompletionChoice] = self.request(prompt, **self.ppl_kwargs)

        ppls = []
        for result, (src, _) in zip(results, batched_inputs):
            tgt_start = max(1, result.logprobs.text_offset.index(len(src)))  # designed for src=''
            tgt_end = len(result.logprobs.text_offset)
            ppl = -sum(result.logprobs.token_logprobs[tgt_start:])
            ppls.append((ppl, tgt_end - tgt_start))
        return ppls

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

        self.prob_kwargs["logprobs"] = len(label_texts)

        if isinstance(logit_bias, list):
            results = [self.request(batched_prompts, logit_bias=lb, **self.prob_kwargs) for lb in logit_bias]
        else:
            results = self.request(batched_prompts, logit_bias=logit_bias, **self.prob_kwargs)

        answers = []
        for result, option_num, label in zip(results, batched_option_nums, label_texts):
            result: Union[CompletionChoice, Choice] = result[0] if isinstance(result, list) else result
            if isinstance(
                result, CompletionChoice
            ) and result.logprobs is not None and result.logprobs.top_logprobs is not None:

                # get the probabilities of each option
                probs = [-9999.] * (option_num * 2)
                top_logprobs = result.logprobs.top_logprobs[0]
                for l, p in top_logprobs.items():
                    text = l.strip()
                    if text in label:
                        probs[label.index(text)] = p

            elif isinstance(result, Choice) and result.logprobs is not None and result.logprobs.content is not None:

                probs = [-9999.] * (option_num * 2)
                top_logprobs = result.logprobs.content[0].top_logprobs
                for top_logprob in top_logprobs:
                    text = top_logprob.token.strip()
                    if text in label:
                        probs[label.index(text)] = top_logprob.logprob

            else:

                # if probabilities are not available, set the probability of the correct answer to 20.0
                # because of logit_bias, the response text will be shown in `label`
                probs = [-9999.] * (option_num * 2)
                text = result.text if isinstance(result, CompletionChoice) else result.message.content
                text = self._option_regex[option_num - 1].findall(text.strip().split("\n")[0])
                if len(text) > 0 and text[-1] in label:
                    probs[label.index(text[-1])] = 20.0

            answers.append(probs)
        return answers
