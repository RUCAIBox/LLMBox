from bisect import bisect_left, bisect_right
from copy import deepcopy
from dataclasses import dataclass
from pprint import pformat
from typing import Callable, List, Literal, Tuple, Union

import tiktoken
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ...model.model_utils.conversation import Conversation, ConversationFormatter


@dataclass
class TokenizerUtilMixin:

    def set_tokenizer(
        self, tokenizer: Union[tiktoken.Encoding, PreTrainedTokenizer, PreTrainedTokenizerFast, None]
    ) -> None:
        self.tokenizer = tokenizer
        if isinstance(tokenizer, tiktoken.Encoding):
            # Encoding.encode_ordinary is slightly faster than Encoding.encode
            self.tokenizer_encode = tokenizer.encode_ordinary
        elif isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            self.tokenizer_encode = tokenizer.encode
        if tokenizer is not None:
            self.tokenizer_decode = tokenizer.decode

    def _apply_normalization(self, conversations: List[Conversation], add_few_shot: bool = False):
        normalization_prompt = getattr(self, "normalization_prompt", None)
        assert normalization_prompt is not None

        def get_normalized_conv_zero_shot(conv: Conversation) -> Conversation:
            new_conv = Conversation.from_chat(user=normalization_prompt, assistant=conv[-1]["content"])
            new_conv.is_normalized = True
            new_conv.set_num_options(conv.num_options)
            new_conv.set_num_shots(0)
            new_conv.set_formatter(conv.formatter, conv.model_evaluation_method, conv.split)
            return new_conv

        def get_normalized_conv_few_shot(conv: Conversation) -> Conversation:
            messages = deepcopy(conv.messages)
            messages[-2]["content"] = normalization_prompt
            return Conversation(
                messages,
                num_turns=conv.num_turns,
                num_options=conv.num_options,
                num_shots=conv.num_shots,
                formatter=conv.formatter,
                model_evaluation_method=conv.model_evaluation_method,
                split=conv.split,
                is_normalized=True,
            )

        get_conv = get_normalized_conv_few_shot if add_few_shot else get_normalized_conv_zero_shot

        normalized_conversations = [get_conv(conv) for conv in conversations]
        conversations.extend(normalized_conversations)

    def prompt_token_nums(self, prompt: str):
        return len(self.tokenizer_encode(prompt))

    def truncate_by_word(
        self,
        words: List[str],
        max_tokens: int,
        side: Literal["left", "right"],
    ) -> Tuple[str, int, int]:
        """Truncate the prompt by word to fit the maximum token length.

        Return:
            - prompt: the truncated prompt
            - real_token_nums: the real token numbers of the truncated prompt
            - word_nums: the number of words in the truncated prompt
        """
        lengths = [0]
        for w in words:
            lengths.append(lengths[-1] + len(w))
        prompt = "".join(words)

        tokens = self.tokenizer_encode(prompt)
        real_token_nums = len(tokens)
        if real_token_nums <= max_tokens:
            return prompt, real_token_nums, len(words)

        st = 0
        ed = len(words)
        if side == "left":
            truncated_raw = self.tokenizer_decode(tokens[-max_tokens:])
            st = bisect_left(lengths, len(prompt) - len(truncated_raw))
        elif side == "right":
            truncated_raw = self.tokenizer_decode(tokens[:max_tokens])
            ed = bisect_right(lengths, len(truncated_raw)) - 1
        prompt = "".join(words[st:ed])
        real_token_nums = self.prompt_token_nums(prompt)
        return prompt, real_token_nums, ed - st

    def _log_instance(self, log: Callable, instance: Conversation, idx: int):
        formatter = getattr(self, "conversation_formatter", None)
        if isinstance(formatter, ConversationFormatter):
            istr = formatter.apply_prompt_template(instance, add_gen_prompt=True)
            log(f"Formatted evaluation instance {idx}\n" + pformat(istr, width=100))
        else:
            for i, seg in enumerate(instance):
                log(f"Formatted evaluation instance {idx} ({seg['role']}_{i})\n" + pformat(seg["content"], width=100))
