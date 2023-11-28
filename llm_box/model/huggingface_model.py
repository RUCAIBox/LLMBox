from logging import getLogger
from typing import Iterator

import torch
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig

from ..utils import ModelArguments
from .model import Model
from .utils import load_llm_and_tokenizer

logger = getLogger(__name__)


class HuggingFaceModel(Model):

    type = "instruction"

    def __init__(self, model_name_or_path: str, args: ModelArguments):
        super().__init__(args)
        self.args = args
        self.model_name_or_path = model_name_or_path

        model, tokenizer = load_llm_and_tokenizer(model_name_or_path, args=self.args)
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        # generation arguments
        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # perplexity arguments
        self.add_start_token = True
        self.model_max_tokens = self.args.max_new_tokens - int(self.add_start_token)
        self.loss_fct = CrossEntropyLoss(reduction="none")

    @staticmethod
    def _subsentences_start_idx(offset_mapping: torch.Tensor) -> Iterator[int]:
        r"""Given offset mapping, return the index of the first token in the encoded sentence of each subsentence. The length of the encoded sentence will be yielded at the end, to ensure that the end index of the last subsentence will be included."""
        for token_idx, (char_st, char_ed) in enumerate(offset_mapping):
            if char_st == 0:
                yield token_idx
        yield len(offset_mapping)

    def get_ppl(self, batched_inputs):

        if not all([tgt.startswith(" ") for _, tgt in batched_inputs]):
            logger.warning(
                f'Target text does not start with a whitespace: ...{batched_inputs[0][0][-10:]}{batched_inputs[0][1][:10]}..."'
            )

        # add_special_tokens=False will still pad the inputs
        batched_encodings = self.tokenizer(
            batched_inputs,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_tensors="pt",
            max_length=self.model_max_tokens,
        ).to(self.device)

        encoded_batch = batched_encodings.input_ids
        attn_mask = batched_encodings.attention_mask

        if self.add_start_token:
            bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
                                             ).to(self.device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat([torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_mask],
                                  dim=1)

        with torch.no_grad():
            out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

        ppls = []
        zipped = zip(out_logits, encoded_batch, attn_mask, batched_encodings.offset_mapping)

        for logits, input_ids, attn_masks, offsets in zipped:
            tgt_st, tgt_ed = list(self._subsentences_start_idx(offsets))[1:3]
            # output logits are shifted by one position
            shift_logits = logits[tgt_st - 1:tgt_ed - 1, :].contiguous()
            shift_labels = input_ids[tgt_st:tgt_ed].contiguous()
            shift_attn_masks = attn_masks[tgt_st:tgt_ed].contiguous()

            perplexity = torch.exp(
                (self.loss_fct(shift_logits, shift_labels) * shift_attn_masks).sum(0) / shift_attn_masks.sum(0)
            )
            ppls.append((perplexity.item(), tgt_ed - tgt_st))

        return ppls

    def generation(self, batched_inputs):
        batched_encodings = self.tokenizer(
            batched_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.model_max_tokens,
        ).to(self.device)

        outputs = self.model.generate(
            **batched_encodings,
            generation_config=self.generation_config,
        )
        answers = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        logger.info(answers)
        return answers
