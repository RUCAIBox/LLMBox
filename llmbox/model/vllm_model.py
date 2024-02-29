from logging import getLogger
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from .model import Model

if TYPE_CHECKING:
    from ..utils import ModelArguments

try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please install vllm by `pip install vllm` to use vllm model. Or you can use huggingface model by `--vllm False`."
    )

logger = getLogger(__name__)


class LabelProcessor:

    def __init__(self, candidate_ids: List[int]) -> List[int]:
        self.candidate_ids = candidate_ids

    def __call__(self, token_ids: List[int], logits_row: torch.Tensor) -> torch.Tensor:
        if len(token_ids) != 0:
            logger.warning("LabelProcessor should be used with max_tokens=1")
        mask = torch.zeros_like(logits_row, dtype=torch.bool)
        mask[self.candidate_ids] = True
        logits_row[~mask] = float("-inf")
        return logits_row


class vllmModel(Model):

    def __init__(self, args: "ModelArguments"):
        super().__init__(args)
        self.args = args

        logger.info(f"Trying to load {args.model_name_or_path} using vllm...")
        self.type = args.model_type
        self.model = LLM(
            model=args.model_name_or_path,
            tokenizer=args.tokenizer_name_or_path,
            tensor_parallel_size=torch.cuda.device_count()
        )
        self.tokenizer = self.model.get_tokenizer()
        self.tokenizer.truncation_side = "left"
        self.tokenizer.model_max_length = min(
            self.model.llm_engine.model_config.max_model_len,
            getattr(args, "max_length") or 1e10
        )

    def set_ppl_args(self, **extra_model_args):
        self.ppl_kwargs = SamplingParams(max_tokens=1, prompt_logprobs=0)

    def get_ppl(self, batched_inputs):
        prompt = [src + tgt for src, tgt in batched_inputs]
        batched_encodings = self.tokenizer(
            prompt, truncation=True, return_offsets_mapping=True, return_attention_mask=False
        )
        results = self.model.generate(prompt_token_ids=batched_encodings.input_ids, sampling_params=self.ppl_kwargs)
        ppls = []
        for result, (src, _), offset in zip(results, batched_inputs, batched_encodings.offset_mapping):
            ppl = [next(iter(r.values())) if r else None for r in result.prompt_logprobs]
            offset = [st for st, ed in offset]
            tgt_start = max(offset.index(len(src)), 1)  # designed for src=''
            tgt_end = len(offset)
            ppl = -sum(ppl[tgt_start:])
            ppls.append((ppl, tgt_end - tgt_start))
        return ppls

    def set_generation_args(self, **extra_model_args):
        generation_kwargs = {}
        for key in [
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "best_of",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "length_penalty",
            "early_stopping",
            "stop",
        ]:
            # ModelArguments > extra_model_args
            value = getattr(self.args, key, None)
            if value is None:
                value = extra_model_args.get(key, None)

            if key == "max_tokens" and value is None:
                value = 1024
            if value is not None:
                generation_kwargs[key] = value
        if generation_kwargs.get("best_of", 0) > 1:
            generation_kwargs["use_beam_search"] = True
        self.generation_kwargs = SamplingParams(**generation_kwargs)

    def generation(self, batched_inputs) -> List[str]:
        results = self.model.generate(batched_inputs, sampling_params=self.generation_kwargs)
        return [r.outputs[0].text for r in results]

    def set_prob_args(self, **extra_model_args):
        self.prob_kwargs = SamplingParams(max_tokens=1, temperature=0)
        self.candidate_ids = extra_model_args.pop("candidate_ids", None)

        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")

    def _set_candidate_ids(self, option_num: int):
        labels = [chr(i + 65) for i in range(option_num)]
        self.word_labels = [self.tokenizer.encode(l, add_special_tokens=False)[0] for l in labels]
        self.token_labels = [self.tokenizer.convert_tokens_to_ids(l) for l in labels]
        return self.word_labels + self.token_labels

    def get_prob(self, batched_inputs: List[Tuple[str, int]]) -> List[List[float]]:
        batched_prompts, batched_option_nums = map(list, zip(*batched_inputs))
        if self.candidate_ids is None:
            max_option_num = max(batched_option_nums)
            candidate_ids = self._set_candidate_ids(max_option_num)
        else:
            candidate_ids = self.candidate_ids
        self.prob_kwargs.logprobs = len(candidate_ids)
        self.prob_kwargs.logits_processors = [LabelProcessor(candidate_ids)]

        results = self.model.generate(
            batched_prompts,
            sampling_params=self.prob_kwargs,
        )
        answers = []
        for result, option_num in zip(results, batched_option_nums):
            if self.candidate_ids is None:
                cur_candidate_ids = self.word_labels[:option_num] + self.token_labels[:option_num]
            else:
                cur_candidate_ids = self.candidate_ids
            prob = torch.tensor([result.outputs[0].logprobs[0][idx] for idx in cur_candidate_ids])
            prob = torch.softmax(prob, dim=0).tolist()
            answers.append(prob)
        return answers
