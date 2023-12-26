from functools import partial
from logging import getLogger
from pprint import pformat
from typing import Iterator, List

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, StoppingCriteria

from ..utils import ModelArguments
from .model import Model
from .utils import load_llm_and_tokenizer, LoggedDict

logger = getLogger(__name__)


class KeyWordsCriteria(StoppingCriteria):

    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequences_should_be_stopped.append(True)
                    break
            sequences_should_be_stopped.append(False)
        return all(sequences_should_be_stopped)


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

        # get the correct max length of the model, which is agnostic to tasks
        max_sequence_length = getattr(self.model.config, "max_sequence_length", None) or 0
        max_position_embeddings = getattr(self.model.config, "max_position_embeddings", None) or 0
        self.model_max_length = self.args.max_sequence_length or max(max_position_embeddings, max_sequence_length)
        if self.model_max_length <= 0:
            raise ValueError(
                f"max_sequence_length is not specified in the model config, and no value is provided in the arguments."
            )
        logger.info(f"model_max_length: {self.model_max_length}")

    @staticmethod
    def _subsentences_start_idx(offset_mapping: torch.Tensor) -> Iterator[int]:
        r"""Given offset mapping, return the index of the first token in the encoded sentence of each subsentence. The length of the encoded sentence will be yielded at the end, to ensure that the end index of the last subsentence will be included."""
        for token_idx, (char_st, char_ed) in enumerate(offset_mapping):
            if char_st == 0:
                yield token_idx
        yield len(offset_mapping)

    def set_ppl_args(self, **kwargs):
        r"""Set the configurations for PPL score calculation. This is useful because different datasets may have different requirements for ppl calculation."""
        self.add_start_token = True
        self.task_max_length = self.model_max_length - int(self.add_start_token)
        self.loss_fct = CrossEntropyLoss(reduction="none")
        logger.info(f"task_max_length: {self.task_max_length}")

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
            max_length=self.task_max_length,
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

    def set_generation_args(self, **kwargs):
        r"""Set the configurations for open-ended generation. This is useful because different datasets may have different requirements for generation.

        Args:
            **kwargs: The generation arguments. It will first be merged with a `GenerationConfig`, and then all unused kwargs are passed as model kwargs. Huggingface-model-specific kwargs are also accepted:

                - `stop_id_sequences` (List[List[int]]): A list of list of token ids. If a list of token ids is found at the end of the generated sequence, the generation will be stopped.
                - `stopping_criteria` (List[StoppingCriteria]): A list of stopping criteria. If any of the stopping criteria is met, the generation will be stopped.
        """
        kwargs = LoggedDict.from_dict(kwargs, logger, "generation_args")
        self.processors = []

        echo = kwargs.pop("echo", False)
        stop_sequences = kwargs.pop("stop_sequences", [])
        stopping_criteria = kwargs.pop("stopping_criteria", [])
        max_new_tokens = kwargs.pop("max_new_tokens", self.args.max_new_tokens) or 512

        self.task_max_length = self.model_max_length - max_new_tokens - 1
        if self.task_max_length <= 0:
            raise ValueError(
                f"`max_new_tokens` is too large ({max_new_tokens}) to fit the capacity of model. Please decrease it to at least {self.model_max_length - 2}."
            )
        logger.info(f"task_max_length: {self.task_max_length}")

        if len(stop_sequences) > 0:
            stop_id_sequences = []
            for sequence in stop_sequences:
                # add a prefix space and manually remove it: https://github.com/huggingface/transformers/issues/26273
                stop_id_sequences.append(self.tokenizer.encode(" " + sequence, add_special_tokens=False)[1:])

            logger.debug(f"stop_id_sequences: {stop_id_sequences}")
            stopping_criteria.append(KeyWordsCriteria(stop_id_sequences))
            self.processors.append(
                partial(
                    self.stop_processor,
                    stop_id_sequences=stop_id_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            )

        if not echo:
            self.processors.append(partial(self.remove_echo_processor, pad_token_id=self.tokenizer.pad_token_id))

        logger.debug(f"extra generation_config: {kwargs}")
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            **kwargs
        )

    def generation(self, batched_inputs) -> List[str]:
        """Generate the response of given question for this batch.

        Returns:
            List(str): The list of generation results.
        """
        # print(batched_inputs)
        batched_encodings = self.tokenizer(
            batched_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            max_length=self.task_max_length,
        ).to(self.device)
        # TODO Python tokenizers doesn't support return_offsets_mapping

        for idx, (lo, li) in enumerate(
            zip(batched_encodings.offset_mapping[:, -1, -1].tolist(), [len(s) for s in batched_inputs])
        ):
            if lo != li:
                overflowed = "..." + batched_inputs[idx][lo:]
                logger.warning(f"Overflowing input during tokenization detected: {pformat(overflowed)}")

        batch_outputs = self.model.generate(
            input_ids=batched_encodings.input_ids,
            attention_mask=batched_encodings.attention_mask,
            generation_config=self.generation_config,
        )

        prompt_len = batched_encodings.input_ids.size(1)
        for seq_idx in range(batch_outputs.size(0)):
            for processor in self.processors:
                processor(seq_idx, prompt_len, batch_outputs)

        answers = self.tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        return answers

    @staticmethod
    def stop_processor(
        seq_idx: int,
        prompt_len: int,
        batch_outputs: Tensor,
        stop_id_sequences: List[List[int]],
        pad_token_id: int,
    ):
        for token_idx in range(prompt_len, batch_outputs[seq_idx, :].size(0)):
            if any(batch_outputs[seq_idx, token_idx:token_idx + len(s)].tolist() == s for s in stop_id_sequences):
                batch_outputs[seq_idx, token_idx:] = pad_token_id
                break

    @staticmethod
    def remove_echo_processor(seq_idx: int, prompt_len: int, batch_outputs: Tensor, pad_token_id: int):
        batch_outputs[seq_idx, :prompt_len] = pad_token_id
