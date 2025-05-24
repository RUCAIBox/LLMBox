import argparse
import os
import sys
import socket
import importlib
from logging import getLogger
from typing import Iterator, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ..model_enum import MEGATRON_ARGS
from ..utils import GenerationArg, ModelArguments, resolve_generation_args
from .model import Model
from .model_utils.conversation import Conversation
from .model_utils.keywords_criteria import KeyWordsCriteria

logger = getLogger(__name__)

_MultiTurnResults = Tuple[str, ...]

def find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a free port assigned by the OS
        return str(s.getsockname()[1])


class MegatronModel(Model):

    model_backend = "megatron"

    model: nn.Module

    _repr = [
        "model_type", "model_backend", "model_max_input",
        "model_max_input_and_output", "multi_turn", "candidate_ids",
        "use_cache"
    ]

    def __init__(self, args: ModelArguments):
        super().__init__(args)

        sys.path.append(args.megatron_path)
        from megatron.training import get_model, get_tokenizer
        from megatron.training.arguments import add_megatron_arguments
        from megatron.training.checkpointing import load_checkpoint
        from megatron.training.initialize import initialize_megatron

        parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                         allow_abbrev=False)
        parser = add_megatron_arguments(parser)
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = find_free_port()

        extra_args = []

        # precision
        if args.torch_dtype == "auto" or args.torch_dtype == "bfloat16":
            extra_args.append("--bf16")
        elif args.torch_dtype == "float16":
            extra_args.append("--fp16")

        # checkpoint step
        if args.megatron_ckpt_step:
            extra_args.extend(["--ckpt-step", str(args.megatron_ckpt_step)])

        assert torch.cuda.device_count() == 1, f"LLMBox currently only support evaluate MegatronModel without TP."

        megatron_args = [
            "--tensor-model-parallel-size",
            str(torch.cuda.device_count()),
            "--pipeline-model-parallel-size",
            "1",
            "--expert-model-parallel-size",
            "1",
            "--load",
            args.model_name_or_path,
            "--micro-batch-size",
            "1",
            "--seed",
            str(args.seed),
            "--use-checkpoint-args",
            "--no-load-rng",
            "--no-load-optim",
            "--exit-on-missing-checkpoint",
            *extra_args
        ]
        megatron_args = parser.parse_args(megatron_args)
        megatron_args.rank = int(os.getenv('RANK', '0'))
        megatron_args.world_size = int(
            os.getenv("WORLD_SIZE", str(torch.cuda.device_count())))

        initialize_megatron(parsed_args=megatron_args)

        # Set up model and load checkpoint
        if args.megatron_model_provider is None:
            if megatron_args.spec == ["megatron.core.models.mamba.mamba_layer_specs", "mamba_stack_spec"]:
                megatron_model_provider = "pretrain_mamba"
            else:
                megatron_model_provider = "pretrain_gpt"
        else:
            megatron_model_provider = args.megatron_model_provider

        model_provider = importlib.import_module(megatron_model_provider).model_provider
        model = get_model(model_provider, wrap_with_ddp=False)
        load_checkpoint(model, None, None)
        model = model[0]
        self.model = model

        self._tokenizer = get_tokenizer()._tokenizer
        self._tokenizer.model_max_length = megatron_args.max_position_embeddings
        self.model_max_input_and_output = self.tokenizer.model_max_length
        self.device = torch.device("cuda")

        self.support_cache = False
        self.support_char_to_token = True

    @property
    def model_max_input(self):
        return self.tokenizer.model_max_length

    @model_max_input.setter
    def model_max_input(self, value):
        self.tokenizer.model_max_length = value

    @property
    def tokenizer(self):
        return self._tokenizer

    def _reload_tokenizer(self):
        pass

    def _remove_tokenizer(self):
        pass

    def set_ppl_args(self, **extra_model_args):
        r"""Set the configurations for PPL score calculation. This is useful because different datasets may have different requirements for ppl calculation."""
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.model_max_input = self.model_max_input_and_output - 1
        self.max_option_tokens = extra_model_args.pop("max_option_tokens", 128)

        extra_model_args.pop("multi_turn", None)  # ignore
        if len(extra_model_args) > 0:
            logger.warning(f"Unused ppl arguments: {extra_model_args}")

    def get_ppl(
        self,
        batched_inputs: List[Tuple[str, ...]],
        use_cache: bool = True,
        exact_match: bool = False,
    ) -> List[Tuple[float, int]]:

        if use_cache and self.use_cache:
            # grouped_prefixes: a list of batched substrings without the last group (target text) with shape [GroupNum - 1, BatchSize]
            *grouped_prefixes, targets = list(map(list, zip(*batched_inputs)))
            cache_level = len(grouped_prefixes)

            # if cache is available, get_ppl_with_cache
            prefix_cache, cached_num = self.cacher.get_cache()
            if prefix_cache is not None and cached_num == cache_level:
                self.cacher.step()
                return self.get_ppl_with_cache(targets, prefix_cache,
                                               exact_match)

            # pass the input without prefix text to the model
            prefix_cache = self.get_cache(
                grouped_prefixes[cached_num],
                prefix_cache,
                save_next_logits=cached_num == cache_level - 1)
            self.cacher.set_cache(prefix_cache)
            self.cacher.step()
            return []

        prompt = ["".join(seq_tuple) for seq_tuple in batched_inputs]

        batched_encodings = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(
                input_ids=batched_encodings["input_ids"],
                attention_mask=batched_encodings["attention_mask"],
                position_ids=None,
            )
            vocab_size = logits.shape[-1]
            shift_logits = logits.detach()[:, :-1].contiguous()
            shift_labels = batched_encodings["input_ids"][:, 1:].contiguous()
            shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
            probs = self.loss_fct(shift_logits.view(-1, vocab_size),
                                  shift_labels.view(-1)).view(
                                      shift_labels.size(0), -1).cpu()

        tgt_starts = [None] * len(batched_inputs)
        if self.tokenizer.is_fast and self.support_char_to_token:
            src_lengths = [len("".join(pg[:-1])) for pg in batched_inputs]
            tgt_starts = [
                batched_encodings.char_to_token(i, l)
                for i, l in enumerate(src_lengths)
            ]
        if any(st is None for st in tgt_starts):
            src_prompts = [
                "".join(pg[:-1]) for pg, st in zip(batched_inputs, tgt_starts)
                if st is None
            ]
            src_batched_encodings = self.tokenizer(src_prompts,
                                                   truncation=True,
                                                   return_attention_mask=False)

            for i, src_ids in zip(
                (i for i, st in enumerate(tgt_starts) if st is None),
                    src_batched_encodings.input_ids,
            ):
                tgt_starts[i] = len(src_ids)
            self.support_char_to_token = False
        ed = len(batched_encodings["input_ids"][0])

        if exact_match:
            ppls = []
            greedy_tokens = torch.argmax(shift_logits, dim=-1)
            for idx, st in enumerate(tgt_starts):
                if greedy_tokens[idx, st - 1:].eq(shift_labels[idx,
                                                               st - 1:]).all():
                    ppl = 0
                else:
                    ppl = probs[idx, st - 1:].sum().item()
                ppls.append((ppl, ed - st))
        else:
            ppls = [(prob[st - 1:].sum().item(), ed - st)
                    for prob, st in zip(probs, tgt_starts)]
        return ppls

    def set_prob_args(self, **extra_model_args):
        self._token_labels = []
        self._word_labels = []
        self.candidate_ids = extra_model_args.pop("candidate_ids", None)

        self.constant_option_num = extra_model_args.pop(
            "constant_option_num", False)

        extra_model_args.pop("multi_turn", None)  # ignore
        if len(extra_model_args) > 0:
            logger.warning(f"Unused prob arguments: {extra_model_args}")

    def _get_label_ids(self, option_num: Optional[int]) -> List[int]:
        """Return the tokenized labels of options."""
        if option_num is not None:
            if len(self._token_labels) < option_num:
                labels = [
                    chr(i + 65)
                    for i in range(len(self._token_labels), option_num)
                ]
                self._word_labels.extend([
                    self.tokenizer.encode(" " + l,
                                          add_special_tokens=False)[-1]
                    for l in labels
                ])
                self._token_labels.extend(
                    [self.tokenizer.convert_tokens_to_ids(l) for l in labels])
            return self._word_labels[:
                                     option_num] + self._token_labels[:
                                                                      option_num]
        else:
            if self.candidate_ids is None:
                raise ValueError(
                    "The candidate_ids must be provided when option_num is None."
                )
            return self.candidate_ids

    def get_prob(self,
                 batched_inputs: List[Tuple[str, int]],
                 use_cache: bool = True) -> List[List[float]]:

        if len(batched_inputs[0]) <= 2:
            batched_prompts, batched_option_nums = map(list,
                                                       zip(*batched_inputs))
        else:
            # batched_groups: a batch of concatenated input strings
            # grouped_prompts: a list of batched substrings with shape [GroupNum, BatchSize]
            *grouped_prompts, batched_option_nums = map(
                list, zip(*batched_inputs))
            batched_prompts = [
                "".join(seq_tuple[:-1]) for seq_tuple in batched_inputs
            ]
            cache_level = len(grouped_prompts) - 1

        if self.use_cache and use_cache:
            # if cache is available, get_prob_with_cache
            prefix_cache, cached_num = self.cacher.get_cache()
            if cached_num == -1:
                self.use_cache = False
            elif prefix_cache is not None and cached_num == cache_level:
                self.cacher.step()
                return self.get_prob_with_cache(grouped_prompts[-1],
                                                batched_option_nums,
                                                prefix_cache)
            else:
                # pass the input without prefix text to the model
                prefix_cache = self.get_cache(grouped_prompts[cached_num],
                                              prefix_cache,
                                              save_next_logits=False)
                self.cacher.set_cache(prefix_cache)
                self.cacher.step()
                return []

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
                attention_mask=batched_encodings["attention_mask"].to(
                    self.device),
                position_ids=None,
            ).detach()[:, -1].contiguous()  # padding_side="left" in tokenizer

            answers = []
            for i, option_num in enumerate(batched_option_nums):
                label_ids = self._get_label_ids(option_num)
                answers.append(
                    torch.softmax(batch_logits[i, label_ids],
                                  dim=-1,
                                  dtype=batch_logits.dtype).tolist())
        return answers

    def set_generation_args(self, **extra_model_args):

        self.multi_turn = extra_model_args.pop("multi_turn", False)
        if self.model_type != "chat" and self.multi_turn:
            raise ValueError(
                "The multi_turn is only available for chat-based model. Please use a chat model and set `--model_type chat`."
            )

        self.stop_id_sequences = []

        def add_stop(value, details: GenerationArg):
            self.stop_id_sequences.extend(
                self._tokenize_postfix(
                    value,  # type: ignore
                    add_dummy_prefix=True,
                    padding=False,
                ))
            return {
                "stopping_criteria":
                [KeyWordsCriteria(self.stop_id_sequences)]
            }

        self.generation_kwargs = resolve_generation_args(
            self.args,
            extra_model_args,
            MEGATRON_ARGS,
            extra_generation_args={
                "stop": add_stop,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            },
        )

        if len(extra_model_args) > 0:
            logger.warning(f"Unused generation arguments: {extra_model_args}")
        return self.generation_kwargs

    def generation(self,
                   batched_inputs: List[Conversation],
                   use_cache: bool = True
                   ) -> Union[List[str], List[_MultiTurnResults]]:
        """Generate the response of given question for this batch.

        Returns:
            List[str]: The list of generation results.
        """
        # batched_inputs: List[Conversation], batched_prompts: List[str] or List[List[str]]
        batched_prompts = [i.to_model_prompt() for i in batched_inputs]
        num_turns = batched_inputs[0].num_turns
        assert all(conv.num_turns == num_turns for conv in batched_inputs)
        if not isinstance(batched_prompts[0], str):
            grouped_prompts = list(map(list, zip(*batched_prompts)))
            cache_level = len(grouped_prompts)

        if use_cache and self.use_cache:
            # if cache is available, generation_with_cache
            prefix_cache, cached_num = self.cacher.get_cache()
            if prefix_cache is not None and cached_num == cache_level - 1:
                self.cacher.step()
                return self.generation_with_cache(grouped_prompts[-1],
                                                  prefix_cache)

            # pass the input without prefix text to the model
            prefix_cache = self.get_cache(grouped_prompts[cached_num],
                                          prefix_cache,
                                          save_token_ids=True)
            self.cacher.set_cache(prefix_cache)
            self.cacher.step()
            return []

        for turn_idx in range(num_turns):
            batched_inputs = self._generation(batched_inputs, turn_idx + 1)

        return [c.get_generation_results() for c in batched_inputs]

    def _generation(
        self,
        batched_inputs: Union[List[str], List[Conversation]],
        max_turns=1,
    ) -> Union[List[str], List[Conversation]]:

        batched_conversations = None
        if isinstance(batched_inputs[0], Conversation):
            # save the original conversation for chat-based model
            batched_conversations = batched_inputs
            batched_inputs_nofilter = [
                conv.to_model_prompt(max_turns=max_turns)
                for conv in batched_conversations
            ]
            # deal with conversations with different number of turns
            batched_inputs = [
                i for i in batched_inputs_nofilter if i is not None
            ]

            def iter_conv() -> Iterator[Conversation]:
                for conv, i in zip(batched_conversations,
                                   batched_inputs_nofilter):
                    if i is not None:
                        yield conv

        batched_encodings = self.tokenizer(
            batched_inputs,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.device)

        batch_outputs = self.model.generate(**batched_encodings,
                                            **self.generation_kwargs)
        for criteria in self.generation_kwargs.get("stopping_criteria", []):
            if isinstance(criteria, KeyWordsCriteria):
                criteria.step()

        max_input_length = batched_encodings["input_ids"].size(1)
        answers = self._process_generation_results(
            batch_outputs[:, max_input_length:])

        if batched_conversations is not None:
            for conv, answer in zip(iter_conv(), answers):
                conv.add_multi_turn(assistant=answer)
            answers = batched_conversations

        return answers

    def _process_generation_results(self,
                                    batch_outputs: torch.Tensor) -> List[str]:
        """Remove the sequences after the `stop_id_sequences` and decode to strings."""
        max_output_length = batch_outputs.size(1)
        if getattr(self, "stop_id_sequences", None) is not None:
            for seq_idx in range(batch_outputs.size(0)):
                for token_idx in range(max_output_length):
                    if any(batch_outputs[seq_idx, token_idx:token_idx +
                                         len(s)].tolist() == s
                           for s in self.stop_id_sequences):
                        batch_outputs[seq_idx,
                                      token_idx:] = self.tokenizer.pad_token_id
                        break

        answers = self.tokenizer.batch_decode(
            batch_outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        return answers
