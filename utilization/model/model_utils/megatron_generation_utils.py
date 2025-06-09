import inspect
import torch
from torch import nn
from typing import Optional, Union, Callable, List

from transformers.cache_utils import Cache, StaticCache
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, MinPLogitsWarper
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerateOutput
from transformers.generation.stopping_criteria import MaxLengthCriteria, StopStringCriteria, EosTokenCriteria
from transformers import PreTrainedTokenizer
from transformers import GenerationMixin


def _get_logits_processor(generation_config: GenerationConfig, logits_processor: Optional[LogitsProcessorList]):
    processors = LogitsProcessorList()

    if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))

    if generation_config.do_sample:
        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            processors.append(TemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            processors.append(
                TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=1)
            )
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            processors.append(
                TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1)
            )
        if generation_config.min_p is not None:
            # Applied after temperature scaling (see https://github.com/ggerganov/llama.cpp/pull/3841#issuecomment-2073826084)
            processors.append(
                MinPLogitsWarper(min_p=generation_config.min_p, min_tokens_to_keep=1)
            )

    if logits_processor is not None:
        processors.extend(logits_processor)  # we do not check duplicate for simplicity
    return processors


def _get_stopping_criteria(generation_config: GenerationConfig, stopping_criteria: Optional[StoppingCriteriaList], tokenizer: Optional[PreTrainedTokenizer]):
    criteria = StoppingCriteriaList()
    if generation_config.max_length is not None:
        max_position_embeddings = getattr(generation_config, "max_position_embeddings", None)
        criteria.append(
            MaxLengthCriteria(
                max_length=generation_config.max_length,
                max_position_embeddings=max_position_embeddings,
            )
        )

    if generation_config.stop_strings is not None:
        if tokenizer is None:
            raise ValueError(
                "There are one or more stop strings, either in the arguments to `generate` or in the "
                "model's generation config, but we could not locate a tokenizer. When generating with "
                "stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`."
            )
        criteria.append(StopStringCriteria(stop_strings=generation_config.stop_strings, tokenizer=tokenizer))
    if generation_config._eos_token_tensor is not None:
        criteria.append(EosTokenCriteria(eos_token_id=generation_config._eos_token_tensor))

    if stopping_criteria is not None:
        criteria.extend(stopping_criteria)  # we do not check duplicate for simplicity
    return criteria


def _prepare_special_tokens(generation_config: GenerationConfig, device: Optional[Union[torch.device, str]] = None):
    # Convert special tokens to tensors
    def _tensor_or_none(token, device=None):
        if token is None:
            return token

        if isinstance(token, torch.Tensor):
            return token.to(device)
        return torch.tensor(token, device=device, dtype=torch.long)

    generation_config._bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
    generation_config._eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
    generation_config._pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
    generation_config._decoder_start_token_tensor = _tensor_or_none(generation_config.decoder_start_token_id, device=device)


def _expand_inputs_for_generation(expand_size: int = 1, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None):
    if input_ids is not None:
        input_ids = input_ids.repeat_interleave(expand_size, dim=0)
    if attention_mask is not None:
        attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
    return input_ids, attention_mask


def _has_unfinished_sequences(this_peer_finished: bool, synced_gpus: bool, device: torch.device) -> bool:
    """
    Returns whether there are still unfinished sequences in the device. The existence of unfinished sequences is
    fed through `this_peer_finished`. ZeRO stage 3-friendly.
    """
    if synced_gpus:
        # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
        # The following logic allows an early break if all peers finished generating their sequence
        this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0, device=device)
        # send 0.0 if we finished, 1.0 otherwise
        torch.distributed.all_reduce(this_peer_finished_flag, op=torch.distributed.ReduceOp.SUM)
        # did all peers finish? the reduced sum will be 0.0 then
        if this_peer_finished_flag.item() == 0.0:
            return False
    elif this_peer_finished:
        return False
    return True


def _sample(
    model,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    past_key_values: Optional[StaticCache] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    while _has_unfinished_sequences(this_peer_finished, synced_gpus = False, device=input_ids.device):
        model_inputs = model.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, attention_mask=attention_mask)

        logits = model(**model_inputs)

        next_token_logits = logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        next_token_scores = logits_processor(input_ids, next_token_logits)

        # token selection
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, next_token_scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del logits

    return input_ids


def gpt_prepare_inputs_for_generation(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[Cache] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """
    Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
    slicing inputs given the existing cache.

    See the forward pass in the model documentation for expected arguments (different models might have different
    requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
    """

    # 1. Handle BC:
    model_inputs = {}

    # 2. Generic cache-dependent input preparation
    if past_key_values is not None:
        model_inputs["past_key_values"] = past_key_values
        inputs_embeds, input_ids = self._cache_dependant_input_preparation(
            input_ids, inputs_embeds, cache_position
        )

    # 3. Prepare base model inputs
    # `clone` calls in this function ensure a consistent stride. See #32227
    model_inputs["input_ids"] = input_ids.clone(memory_format=torch.contiguous_format)

    # 4. Create missing `position_ids` on the fly
    if (
        attention_mask is not None
        and kwargs.get("position_ids") is None
    ):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        kwargs["position_ids"] = position_ids  # placed in kwargs for further processing (see below)

    # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
    for model_input_name in ["position_ids", "token_type_ids"]:
        model_input = kwargs.get(model_input_name)
        if model_input is not None:
            if past_key_values is not None:
                current_input_length = model_inputs["input_ids"].shape[1]
                model_input = model_input[:, -current_input_length:]
                model_input = model_input.clone(memory_format=torch.contiguous_format)
            model_inputs[model_input_name] = model_input

    # 6. Create 4D attention mask is we are using a compilable cache (important for performant compiled forward
    # pass)
    if (
        isinstance(past_key_values, Cache)
        and past_key_values.is_compileable
        and attention_mask is not None
        and attention_mask.ndim == 2
    ):
        batch_size, sequence_length = model_inputs["input_ids"].shape[:2]

        # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
        # the 4D causal mask exists, it should be present in the base model (XXXModel class) or in its decoder.
        base_model = getattr(self, self.base_model_prefix, self)
        decoder = base_model.get_decoder() if hasattr(base_model, "get_decoder") else None
        causal_mask_creation_function = getattr(
            base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
        )
        if causal_mask_creation_function is None and decoder is not None:  # it may be in the decoder
            causal_mask_creation_function = getattr(
                decoder, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
        if causal_mask_creation_function is None:  # can't be found
            print(
                f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                "writing code, see Llama for an example implementation. If you're a user, please report this "
                "issue on GitHub."
            )
        else:
            attention_mask = causal_mask_creation_function(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.dtype,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )
    if attention_mask is not None:
        model_inputs["attention_mask"] = attention_mask

    # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
    for key, value in kwargs.items():
        if key not in model_inputs:
            model_inputs[key] = value

    # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
    model_inputs.pop("labels", None)
    return model_inputs


@torch.no_grad()
def megatron_generate(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    past_key_values: Optional[StaticCache] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> Union[GenerateOutput, torch.LongTensor]:

    assert generation_config is not None, "generation_config must be provided for MegatronModel generation."

    # 3. Define model inputs
    device = input_ids.device
    _prepare_special_tokens(generation_config, device=device)

    # 8. determine generation mode
    generation_mode = generation_config.get_generation_mode()

    # 9. prepare logits processors and stopping criteria
    prepared_logits_processor = _get_logits_processor(generation_config, logits_processor)
    prepared_stopping_criteria = _get_stopping_criteria(generation_config, stopping_criteria, tokenizer)

    # 10. go into different generation modes
    if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, attention_mask = _expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
        result = _sample(
            self,
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
    else:
        raise NotImplementedError(f"Generation mode {generation_mode} is not implemented for MegatronModel.")

    return result
