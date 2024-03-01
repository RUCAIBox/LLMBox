from logging import getLogger
from typing import Iterator, List, Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from ..utils import ModelArguments
from .model import Model

logger = getLogger(__name__)


def load_hf_model(args: ModelArguments) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    logger.info(f"Loading {args.model_name_or_path} using Hugging Face Transformers...")

    model_kwargs = dict(
        torch_dtype=torch.float16,
        device_map=args.device_map,
    )

    if args.flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if hasattr(args, 'bnb_config') and args.bnb_config:
        model_kwargs['quantization_config'] = args.bnb_config

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs).eval()
    except (TypeError, ImportError, ValueError) as e:
        if "attn_implementation" in str(e) or "flash att" in str(e).lower().replace("_", " "):
            logger.warning(
                f"Cannot set `attn_implementation` for {args.model_name_or_path}: {e}. Set `flash_attention` to False."
            )
            args.flash_attention = False
            model_kwargs.pop("attn_implementation")
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs).eval()
        else:
            raise e

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path, use_fast=True, padding_side="left", truncation_side="left", add_eos_token=False
    )

    # TODO: [Important]!!! check for each tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # TODO: [Important]!!! check for each tokenizer
    max_length = min(getattr(tokenizer, "tokenizer_model_max_length", 1e10), getattr(args, "max_length") or 1e10)
    for key in [
        "max_sequence_length",
        "max_position_embeddings",
        "model_max_length",
        "seq_length",
        "seq_len",
        "n_positions",
        "max_seq_len",
        "max_seq_length",
    ]:
        max_length = min(max_length, getattr(model.config, key, 1e10))
    if not max_length or max_length >= 1e10:
        max_length = 2048
        logger.warning(
            f"Cannot specify model's maximum length according to `args` or model config. Set to 2048 by default."
        )

    tokenizer.model_max_length = max_length
    logger.debug(f"Model: {model}\nTokenizer: {tokenizer}")
    return model, tokenizer


class HuggingFaceModel(Model):

    def __init__(self, args: ModelArguments):
        super().__init__(args)
        self.args = args
        self.type = args.model_type
        if self.type not in {"base", "instruction"}:
            raise ValueError(
                f"Invalid model type: {self.type}. Please use `--model_type` to specify the"
                " model type, which can be chosen from `base` and `instruction`."
            )

        self.model, self.tokenizer = load_hf_model(args)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _subsentences_start_idx(offset_mapping: torch.Tensor) -> Iterator[int]:
        r"""Given offset mapping, return the index of the first token in the encoded sentence of each subsentence. The length of the encoded sentence will be yielded at the end, to ensure that the end index of the last subsentence will be included."""
        for token_idx, (char_st, char_ed) in enumerate(offset_mapping):
            if char_st == 0:
                yield token_idx
        yield len(offset_mapping)

    def set_ppl_args(self, **extra_model_args):
        r"""Set the configurations for PPL score calculation. This is useful because different datasets may have different requirements for ppl calculation."""
        self.loss_fct = CrossEntropyLoss(reduction="none")
        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")

    def get_ppl(self, batched_inputs: List[Tuple[str, str]]) -> List[float]:
        prompt = [src + tgt for src, tgt in batched_inputs]

        batched_encodings = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(
                input_ids=batched_encodings["input_ids"], attention_mask=batched_encodings["attention_mask"]
            ).logits
            shift_logits = logits.detach()[:, :-1].contiguous()
            shift_labels = batched_encodings["input_ids"][:, 1:].contiguous()
            shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
            probs = self.loss_fct(shift_logits.view(-1, self.model.config.vocab_size),
                                  shift_labels.view(-1)).view(shift_labels.size(0), -1).cpu()

        ppls = []
        for prob, (src, _), offset, attention_mask in zip(
            probs, batched_inputs, batched_encodings.offset_mapping, batched_encodings.attention_mask
        ):
            ppl = [None] + prob.tolist()
            offset = [st for st, ed in offset]
            tgt_start = max(
                offset.index(len(src)),
                attention_mask.nonzero()[0][0].item() + 1
            )  # designed for src!='' and src=''
            tgt_end = len(offset)
            ppl = sum(ppl[tgt_start:])
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
            "repetition_penalty",
            "length_penalty",
            "early_stopping",
            "no_repeat_ngram_size",
        ]:
            # ModelArguments > extra_model_args
            value = getattr(self.args, key, None)
            if value is None:
                value = extra_model_args.pop(key, None)
            if key == "max_tokens" and value is None:
                value = 1024
            if value is not None:
                if key == "max_tokens":
                    generation_kwargs["max_new_tokens"] = value
                elif key == "best_of":
                    generation_kwargs["num_beams"] = value
                elif key == "temperature":
                    if value > 0:
                        generation_kwargs["temperature"] = value
                        generation_kwargs["do_sample"] = True
                    else:
                        generation_kwargs["do_sample"] = False
                else:
                    generation_kwargs[key] = value

        generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        self.generation_kwargs = generation_kwargs
        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")

    def generation(self, batched_inputs: List[str]) -> List[str]:
        """Generate the response of given question for this batch.

        Returns:
            List[str]: The list of generation results.
        """

        batched_encodings = self.tokenizer(
            batched_inputs,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)

        batch_outputs = self.model.generate(**batched_encodings, **self.generation_kwargs)
        max_input_length = batched_encodings["input_ids"].size(1)
        batch_outputs = batch_outputs[:, max_input_length:]
        answers = self.tokenizer.batch_decode(
            batch_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return answers

    def set_prob_args(self, **extra_model_args):
        self._token_labels = []
        self._word_labels = []
        self._candidate_ids = extra_model_args.pop("candidate_ids", None)

        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")

    def _get_label_ids(self, option_num: Optional[int]) -> List[int]:
        """Return the tokenized labels of options."""
        if option_num is not None:
            if len(self._token_labels) < option_num:
                labels = [chr(i + 65) for i in range(len(self._token_labels), option_num)]
                self._word_labels.extend([self.tokenizer.encode(l, add_special_tokens=False)[0] for l in labels])
                self._token_labels.extend([self.tokenizer.convert_tokens_to_ids(l) for l in labels])
            return self._word_labels[:option_num] + self._token_labels[:option_num]
        else:
            if self._candidate_ids is None:
                raise ValueError("The candidate_ids must be provided when option_num is None.")
            return self._candidate_ids

    def get_prob(self, batched_inputs: List[Tuple[str, int]]) -> List[List[float]]:
        batched_prompts, batched_option_nums = map(list, zip(*batched_inputs))
        batched_encodings = self.tokenizer(
            batched_prompts,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            batch_logits = self.model(
                input_ids=batched_encodings["input_ids"].to(self.device),
                attention_mask=batched_encodings["attention_mask"].to(self.device),
            ).logits.detach()[:, -1].contiguous()  # padding_side="left" in tokenizer

            answers = []
            for i, option_num in enumerate(batched_option_nums):
                label_ids = self._get_label_ids(option_num)
                answers.append(torch.softmax(batch_logits[i, label_ids], dim=-1, dtype=torch.float32).tolist())
        return answers
