from functools import partial
from logging import getLogger
from pprint import pformat
from typing import Iterator, List, Union, Tuple

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast


from ..utils import ModelArguments
from .model import Model

logger = getLogger(__name__)

def load_hf_model(
    args: ModelArguments
) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    
    logger.info(f"Trying to load {args.model_name_or_path} using Hugging Face Transformers...")

    model_kwargs = dict(
        torch_dtype=torch.float16,
        device_map=args.device_map,
    )

    if args.flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True, padding_side='left', add_eos_token=False)

    # TODO: [Important]!!! check for each tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    # TODO: [Important]!!! check for each tokenizer
    max_length = min(getattr(tokenizer, "tokenizer_model_max_length", 1e10), getattr(args, 'max_length') or 1e10)
    for key in ["max_sequence_length", "max_position_embeddings", "model_max_length", "seq_length", "seq_len", "n_positions", "max_seq_len", "max_seq_length"]:
        max_length = min(max_length, getattr(model.config, key, 1e10))
    if not max_length or max_length >= 1e10:
        max_length = 2048
        logger.warning(f"Cannot specify model's maximum length according to `args` or model config. Set to 2048 by default.")
    
    tokenizer.model_max_length = max_length
    return model, tokenizer


class HuggingFaceModel(Model):

    def __init__(self, args: ModelArguments):
        super().__init__(args)
        self.args = args
        self.type = args.model_type
        self.model, self.tokenizer = load_hf_model(args)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def _subsentences_start_idx(offset_mapping: torch.Tensor) -> Iterator[int]:
        r"""Given offset mapping, return the index of the first token in the encoded sentence of each subsentence. The length of the encoded sentence will be yielded at the end, to ensure that the end index of the last subsentence will be included."""
        for token_idx, (char_st, char_ed) in enumerate(offset_mapping):
            if char_st == 0:
                yield token_idx
        yield len(offset_mapping)

    def set_ppl_args(self, **kwargs):
        r"""Set the configurations for PPL score calculation. This is useful because different datasets may have different requirements for ppl calculation."""
        self.loss_fct = CrossEntropyLoss(reduction="none")

    def get_ppl(self, batched_inputs):
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
            logits = self.model(input_ids=batched_encodings['input_ids'], attention_mask=batched_encodings['attention_mask']).logits
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = batched_encodings['input_ids'][:, 1:].contiguous()
            shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
            probs = self.loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1)).view(shift_labels.size(0), -1)

        ppls = []
        for prob, (src, _), offset, attention_mask in zip(probs, batched_inputs, batched_encodings.offset_mapping, batched_encodings.attention_mask):
            ppl = [None] + prob.tolist()
            offset = [st for st, ed in offset]
            tgt_start = max(offset.index(len(src)), attention_mask.nonzero()[0][0].item() + 1) # designed for src!='' and src=''
            tgt_end = len(offset)
            ppl = sum(ppl[tgt_start:])
            ppls.append((ppl, tgt_end - tgt_start))
        return ppls

    def set_generation_args(self, **kwargs):
        generation_kwargs = {}
        for key in ['temperature', 'top_p', 'top_k', 'max_tokens', 'best_of', 'repetition_penalty', 'length_penalty', 'early_stopping', 'no_repeat_ngram_size']:
            value = getattr(self.args, key) if getattr(self.args, key, None) is not None else kwargs.get(key, None)
            if key == 'max_tokens' and value is None:
                value = 1024
            if value is not None:
                if key == 'max_tokens':
                    generation_kwargs['max_new_tokens'] = value
                elif key == 'best_of':
                    generation_kwargs['num_beams'] = value
                elif key == 'temperature':
                    if value > 0:
                        generation_kwargs['temperature'] = value
                        generation_kwargs['do_sample'] = True
                    else:
                        generation_kwargs['do_sample'] = False
                else:
                    generation_kwargs[key] = value
        
        generation_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        generation_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        self.generation_kwargs = generation_kwargs

    def generation(self, batched_inputs) -> List[str]:
        """Generate the response of given question for this batch.

        Returns:
            List(str): The list of generation results.
        """

        batched_encodings = self.tokenizer(
            batched_inputs,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)

        batch_outputs = self.model.generate(**batched_encodings, **self.generation_kwargs)
        max_input_length = batched_encodings['input_ids'].size(1)
        batch_outputs = batch_outputs[:, max_input_length:]
        answers = self.tokenizer.batch_decode(batch_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return answers
