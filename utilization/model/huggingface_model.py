from bisect import bisect_left
from logging import getLogger
from typing import Any, Iterator, List, Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines.conversational import Conversation
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from ..utils import ModelArguments
from .model import Model
from .model_utils import KeyWordsCriteria

logger = getLogger(__name__)


def load_hf_model(args: ModelArguments) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    logger.info(f"Loading {args.model_name_or_path} using Hugging Face Transformers...")

    # https://github.com/meta-llama/llama/issues/380#issuecomment-1656714118
    model_kwargs = dict(
        torch_dtype=getattr(torch, args.torch_dtype),
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        trust_remote_code=True
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
        args.tokenizer_name_or_path,
        use_fast=True,
        padding_side="left",
        truncation_side="left",
        add_eos_token=False,
        trust_remote_code=True
    )

    # TODO: [Important]!!! check for each tokenizer
    if tokenizer.pad_token is None:
        if "llama2" in args.model_name_or_path.lower().replace("_", "").replace("-", ""):
            # https://github.com/meta-llama/llama/issues/380#issuecomment-1729077205
            tokenizer.pad_token = tokenizer.bos_token
        else:
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

    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

    _repr = ["type", "system_prompt", "multi_turn", "candidate_ids"]

    def __init__(self, args: ModelArguments):
        super().__init__(args)
        self.args = args
        self.type = args.model_type
        if self.type not in {"base", "instruction", "chat"}:
            raise ValueError(
                f"Invalid model type: {self.type}. Please use `--model_type` to specify the"
                " model type, which can be chosen from `base` and `instruction`."
            )

        self.model, self.tokenizer = load_hf_model(args)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.system_prompt = self.args.system_prompt
        self.chat_template = self.args.chat_template
        if self.system_prompt:
            self._system = lambda: Conversation([{"role": "system", "content": self.system_prompt}])
        else:
            self._system = lambda: Conversation()
        if self.chat_template is not None:
            self.tokenizer.chat_template = self.chat_template

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
            logger.warning(f"Unused ppl arguments: {extra_model_args}")

    def get_ppl(self, batched_inputs: List[Tuple[str, str]]) -> List[float]:
        prompt = [src + tgt for src, tgt in batched_inputs]
        return_offsets_mapping = True if isinstance(self.tokenizer, PreTrainedTokenizerFast) else False
        batched_encodings = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_offsets_mapping=return_offsets_mapping,
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
        tgt_st_eds = []
        if return_offsets_mapping:
            for (src, _), offset, attention_mask in zip(
                batched_inputs, batched_encodings.offset_mapping, batched_encodings.attention_mask
            ):
                offset = [
                    offset[i][0] if i == 0 or offset[i][0] == offset[i - 1][1] else offset[i][0] - 1
                    for i in range(len(offset))
                ]
                tgt_start = max(
                    bisect_left(offset, len(src)),
                    attention_mask.nonzero()[0][0].item() + 1
                )  # designed for src!='' and src=''
                tgt_end = len(offset)
                tgt_st_eds.append((tgt_start, tgt_end))
        else:
            src_prompt = [src for src, _ in batched_inputs]
            src_batched_encodings = self.tokenizer(src_prompt, truncation=True, return_attention_mask=False)
            for src_input_ids, input_ids in zip(src_batched_encodings.input_ids, batched_encodings.input_ids):
                tgt_st_eds.append((len(src_input_ids), len(input_ids)))

        for prob, (tgt_start, tgt_end) in zip(probs, tgt_st_eds):
            ppl = [None] + prob.tolist()
            ppl = sum(ppl[tgt_start:])
            ppls.append((ppl, tgt_end - tgt_start))
        return ppls

    def _tokenize_postfix(
        self,
        batched_inputs: List[str],
        prefix_cache: Optional[Any] = None,
        device: Optional[str] = None,
        add_dummy_prefix: bool = False,
        dummy_prefix: str = "text ",
        padding: bool = True,
    ) -> List[List[int]]:
        """Tokenize the inputs as postfix. If `prefix_cache` is provided, the attention_mask of the prefix will be concatenated with the input attention_mask and the position_ids of the input will be calculated based on the length of the prefix.

        Args:
            batched_inputs (`List[str]`): Batched inputs to be tokenized as postfix.
            prefix_cache (`Optional[SequenceCache]`, optional): The SequenceCache of prefix. Defaults to None.
            device (`Optional[str]`, optional): Target device of returned tensors. Defaults to None.
            add_prefix (`bool`, optional): If no `prefix_cache` is provided, use this to add a dummy prefix and remove it after tokenization. Defaults to False.
            padding (`bool`, optional): Whether to pad the sequence (to right) and return in tensor. Defaults to True.

        Returns:
            - If `padding` is True:
                `_PostfixEncoding`: Encoding of postfix with padding.
            - If `padding` is False:
                `List[List[int]]`: A list of tokenized inputs without padding.
        """

        batch_size = len(batched_inputs)
        _to = dict(dtype=torch.long, device=self.device)
        if device is not None:
            _to["device"] = torch.device(device)

        # tokenize the postfix like a postfix. this is useful to handle tokenizers like llama
        if prefix_cache is not None and len(prefix_cache.last_tokens) == batch_size:
            batched_inputs = [l + p for l, p in zip(prefix_cache.last_tokens, batched_inputs)]
        elif add_dummy_prefix:
            batched_inputs = [dummy_prefix + p for p in batched_inputs]

        # use the same tokenizer, but different padding strategy
        batched_encodings = self.tokenizer(batched_inputs)

        if self.tokenizer.is_fast:

            def char_to_token(i, char_index):
                return batched_encodings.char_to_token(i, char_index)
        else:

            def char_to_token(i, char_index):
                return len(self.tokenizer(batched_inputs[i][:char_index]).input_ids)

        # remove the prefix from the input_ids and get the batched_ids for postfix
        if prefix_cache is not None and prefix_cache.last_tokens is not None:
            ids_slice = [
                slice(char_to_token(i, len(l)), self.tokenizer.model_max_length)
                for i, l in enumerate(prefix_cache.last_tokens)
            ]
        elif add_dummy_prefix:
            char_index = len(dummy_prefix)
            ids_slice = [
                slice(char_to_token(i, char_index), self.tokenizer.model_max_length) for i in range(batch_size)
            ]
        else:
            ids_slice = [slice(0, self.tokenizer.model_max_length)] * batch_size
        batched_ids = [i[slc] for i, slc in zip(batched_encodings["input_ids"], ids_slice)]
        input_lengths = [len(seq) for seq in batched_ids]
        max_input_len = max(input_lengths)
        if not padding:
            return batched_ids

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
            "stop",
        ]:
            # ModelArguments (cmd) > extra_model_args > ModelArguments (default)
            if not self.args.passed_in_commandline(key):
                value = extra_model_args.pop(key, None)
            if value is None:
                value = getattr(self.args, key, None)

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
                elif key == "stop":
                    self.stop_id_sequences = self._tokenize_postfix(value, add_dummy_prefix=True, padding=False)
                    generation_kwargs["stopping_criteria"] = [KeyWordsCriteria(self.stop_id_sequences)]
                else:
                    generation_kwargs[key] = value

        generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        self.multi_turn = extra_model_args.pop("multi_turn", False)

        if self.type != "chat":
            if self.multi_turn:
                raise ValueError(
                    "The multi_turn is only available for chat-based model. Please use a chat model and set `--model_type chat`."
                )

        self.generation_kwargs = generation_kwargs
        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")
        return self.generation_kwargs

    def generation(self, batched_inputs: List[str]) -> Union[List[str], List[Tuple[str, ...]]]:
        """Generate the response of given question for this batch.

        Returns:
            List[str]: The list of generation results.
        """

        if self.type == "chat" and self.multi_turn:
            batched_mt_inputs = [s.split("__SEPARATOR__") for s in batched_inputs]
            return self._multi_turn_generation(batched_mt_inputs)
        else:
            return self._generation(batched_inputs)

    def _multi_turn_generation(self, batched_mt_inputs: List[List[str]]) -> List[Tuple[str, ...]]:
        """Multi-turn generation

        Args:
            batched_mt_inputs (List[List[str]]): [Batch Turn]

        Returns:
            List[Tuple[str, ...]]: [Batch Turn]
        """
        batch_size = len(batched_mt_inputs)
        max_turn = max(len(turns) for turns in batched_mt_inputs)

        # generate the first turn (with system prompt)
        history_conversations = [self._system() for _ in range(batch_size)]
        for conv, turn in zip(history_conversations, batched_mt_inputs):
            conv.add_message({"role": "user", "content": turn[0]})

        answers = self._generation(history_conversations)
        responses = [[conv[-1]["content"]] for conv in answers]

        for turn_idx in range(1, max_turn):
            cur_turn = [mt_input[turn_idx] if len(mt_input) > turn_idx else None for mt_input in batched_mt_inputs]
            next_batch = []
            for conv, turn in zip(history_conversations, cur_turn):
                if turn is not None:
                    conv.add_message({"role": "user", "content": turn})
                    next_batch.append(conv)

            answers = self._generation(next_batch)

            for idx, (r, turn) in enumerate(zip(responses, cur_turn)):
                if turn is not None:
                    conv = answers.pop(0)
                    r.append(conv[-1]["content"])
                    history_conversations[idx] = conv

        return [tuple(r) for r in responses]

    def _generation(self, batched_inputs: Union[List[str], List[Conversation]]) -> Union[List[str], List[Conversation]]:

        batched_conversations = None
        if self.type == "chat" and isinstance(batched_inputs[0], Conversation):
            batched_conversations = batched_inputs
            batched_inputs = [
                self.tokenizer.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=True,
                    truncation=True,
                ) for conv in batched_conversations
            ]

        batched_encodings = self.tokenizer(
            batched_inputs,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.device)

        batch_outputs = self.model.generate(**batched_encodings, **self.generation_kwargs)
        for criteria in self.generation_kwargs.get("stopping_criteria", []):
            if isinstance(criteria, KeyWordsCriteria):
                criteria.step()

        max_input_length = batched_encodings["input_ids"].size(1)
        answers = self._process_generation_results(batch_outputs[:, max_input_length:])

        if self.type == "chat" and batched_conversations is not None:
            for conv, answer in zip(batched_conversations, answers):
                conv.add_message({"role": "assistant", "content": answer})
            answers = batched_conversations

        return answers

    def _process_generation_results(self, batch_outputs: torch.Tensor) -> List[str]:
        """Remove the sequences after the `stop_id_sequences` and decode to strings."""
        max_output_length = batch_outputs.size(1)
        if getattr(self, "stop_id_sequences", None) is not None:
            for seq_idx in range(batch_outputs.size(0)):
                for token_idx in range(max_output_length):
                    if any(
                        batch_outputs[seq_idx, token_idx:token_idx + len(s)].tolist() == s
                        for s in self.stop_id_sequences
                    ):
                        batch_outputs[seq_idx, token_idx:] = self.tokenizer.pad_token_id
                        break

        answers = self.tokenizer.batch_decode(
            batch_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return answers

    def set_prob_args(self, **extra_model_args):
        self._token_labels = []
        self._word_labels = []
        self.candidate_ids = extra_model_args.pop("candidate_ids", None)

        if len(extra_model_args) > 0:
            logger.warning(f"Unused prob arguments: {extra_model_args}")

    def _get_label_ids(self, option_num: Optional[int]) -> List[int]:
        """Return the tokenized labels of options."""
        if option_num is not None:
            if len(self._token_labels) < option_num:
                labels = [chr(i + 65) for i in range(len(self._token_labels), option_num)]
                self._word_labels.extend([self.tokenizer.encode(l, add_special_tokens=False)[0] for l in labels])
                self._token_labels.extend([self.tokenizer.convert_tokens_to_ids(l) for l in labels])
            return self._word_labels[:option_num] + self._token_labels[:option_num]
        else:
            if self.candidate_ids is None:
                raise ValueError("The candidate_ids must be provided when option_num is None.")
            return self.candidate_ids

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
