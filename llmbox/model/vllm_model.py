import torch

from logging import getLogger
from typing import List

from ..utils import ModelArguments
from .model import Model

try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please install vllm by `pip install vllm` to use vllm model. Or you can use huggingface model by `--vllm False`."
    )

logger = getLogger(__name__)


class vllmModel(Model):

    def __init__(self, args: ModelArguments):
        super().__init__(args)
        self.args = args

        logger.info(f"Loading {args.model_name_or_path} using vllm...")
        self.type = args.model_type
        self.model = LLM(model=args.model_name_or_path, tokenizer=args.tokenizer_name_or_path, tensor_parallel_size=torch.cuda.device_count())
        self.tokenizer = self.model.get_tokenizer()
        self.tokenizer.model_max_length = min(
            self.model.llm_engine.model_config.max_model_len,
            getattr(args, "max_length") or 1e10
        )

    def set_ppl_args(self, **kwargs):
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

    def set_generation_args(self, **kwargs):
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
            value = getattr(self.args, key) if getattr(self.args, key, None) is not None else kwargs.get(key, None)
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
