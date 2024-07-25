from copy import deepcopy
from functools import lru_cache
from logging import getLogger
from pprint import pformat
from typing import Dict, Iterator, List, Literal, NewType, Optional, Tuple, Union

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

import copy

from ...chat_templates import DEFAULT_CHAT_CONFIGS, DEFAULT_CHAT_TEMPLATE, add_space, smart_space

# legacy types
NumOptions = NewType("NumOptions", int)

PPLInput = NewType("PPLInput", List[Tuple[str, str]])
PPLInputSplited = NewType("PPLInputSplited", List[Tuple[str, ...]])

ProbInput = NewType("ProbInput", List[Tuple[str, NumOptions]])
ProbInputSplited = NewType(
    "ProbInputSplited", List[Union[Tuple[str, str, NumOptions], Tuple[str, str, str, NumOptions]]]
)

GenInput = NewType(
    "GenInput",
    List[str],
)
GenInputSplited = NewType(
    "GenInputSplited",
    List[Union[Tuple[str, str], Tuple[str, str]]],
)

logger = getLogger(__name__)


@lru_cache
def _compile_jinja_template(chat_template):

    def raise_exception(message):
        raise TemplateError(message)

    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.do"])
    jinja_env.globals["raise_exception"] = raise_exception
    jinja_env.filters['smart_space'] = smart_space
    return jinja_env.from_string(chat_template)


class ConversationFormatter:

    def __init__(
        self,
        chat_config: Dict[str, str],
        chat_template: str,
        special_tokens_map: Optional[Dict[str, Union[str, List[str]]]] = None,
    ):
        self.default_stop = chat_config.pop("default_stop", [])
        self.auto_leading_space = chat_config.pop("auto_leading_space", True)
        self.final_lstrip = chat_config.pop("final_lstrip", True)
        self.final_rstrip = chat_config.pop("final_rstrip", True)
        self.merge_system_to_user = chat_config.pop("merge_system_to_user", False)
        self.system_user_sep = chat_config.pop("system_user_sep", "\n")

        # api model does not need bos_token
        if "bos_token" not in chat_config:
            chat_config["bos_token"] = ""

        self.sequences = chat_config
        self.chat_template = chat_template
        self.special_tokens_map = special_tokens_map or {}

    @classmethod
    def from_chat_template(
        cls,
        chat_template: Optional[str],
        special_tokens_map: Optional[Dict[str, Union[str, List[str]]]] = None,
    ) -> "ConversationFormatter":
        if chat_template is None:
            chat_template = "base"

        if chat_template in DEFAULT_CHAT_CONFIGS:
            chat_config = copy.deepcopy(DEFAULT_CHAT_CONFIGS[chat_template])
            chat_config = DEFAULT_CHAT_CONFIGS[chat_template]
            chat_template = DEFAULT_CHAT_TEMPLATE
        else:
            chat_config = {"final_rstrip": False}

        if not isinstance(chat_config, dict):
            chat_config = {}
            chat_template = chat_config

        if special_tokens_map is not None:
            for key, value in special_tokens_map.items():
                if key not in chat_config:
                    chat_config[key] = value

        return cls(chat_config=chat_config, chat_template=chat_template, special_tokens_map=special_tokens_map)

    @staticmethod
    def _apply_prompt_template(
        conversation: Union["Conversation", List[Dict[str, str]], List["Conversation"]],
        prompt_template: str,
        add_gen_prompt: bool,
        remove_gen_prompt: bool,
        final_lstrip: bool,
        final_rstrip: bool,
        auto_leading_space: bool,
        sequences: Optional[Dict[str, str]],
        special_tokens_map: Dict[str, Union[str, List[str]]],
    ) -> str:
        if isinstance(conversation, list):
            if len(conversation) == 0:
                return ""
            if hasattr(conversation[0], "messages"):
                messages = [msg for conv in conversation for msg in conv.messages]
            else:
                messages = conversation
        elif hasattr(conversation, "messages"):
            messages = conversation.messages
        else:
            raise ValueError("Invalid conversation format")

        add_gen_prompt = add_gen_prompt and messages[-1]["role"] != "assistant"

        compiled_template = _compile_jinja_template(prompt_template)

        try:
            rendered = compiled_template.render(
                messages=messages,
                auto_leading_space=auto_leading_space,
                add_gen_prompt=add_gen_prompt,
                seq=sequences,
                **special_tokens_map,
            )
        except TemplateError as e:
            logger.error(f"Error in applying prompt template: {prompt_template}\nmessages={messages}\nseq={sequences}")
            raise e
        if final_lstrip:
            rendered = rendered.lstrip()
        if final_rstrip:
            rendered = rendered.rstrip()
        if remove_gen_prompt and messages[0]["role"] == "assistant" and len(messages) == 1:
            if rendered.startswith(sequences['assistant_start']):
                rendered = rendered[len(sequences['assistant_start']):]
            if rendered.endswith(sequences['assistant_end']):
                rendered = rendered[:-len(sequences['assistant_end'])]

        return rendered

    def apply_prompt_template(
        self,
        conversation: Union["Conversation", List[Dict[str, str]], List["Conversation"]],
        *,
        add_gen_prompt: bool = False,
        remove_gen_prompt: bool = False,
        final_lstrip: Optional[bool] = None,
        final_rstrip: Optional[bool] = None,
    ) -> str:

        if final_lstrip is None:
            final_lstrip = self.final_lstrip

        if final_rstrip is None:
            final_rstrip = self.final_rstrip

        return self._apply_prompt_template(
            conversation=conversation,
            prompt_template=self.chat_template,
            add_gen_prompt=add_gen_prompt,
            remove_gen_prompt=remove_gen_prompt,
            auto_leading_space=self.auto_leading_space,
            final_lstrip=final_lstrip,
            final_rstrip=final_rstrip,
            sequences=self.sequences,
            special_tokens_map=self.special_tokens_map,
        )

    def _get_segs(self, conversations: List["Conversation"], max_turns: int = 1) -> Iterator[tuple]:
        seg_num = None
        kwargs = {"final_rstrip": False, "final_lstrip": False}
        for conv in conversations:
            system = self.apply_prompt_template(conv.get_segs("system"), **kwargs)
            examples = self.apply_prompt_template(conv.get_segs("examples"), **kwargs)
            source = self.apply_prompt_template(
                conv.get_segs("source")[:max_turns * 2 - 1], add_gen_prompt=True, **kwargs
            )
            target = self.apply_prompt_template(conv.get_segs("target"), remove_gen_prompt=True, **kwargs)
            result = ()
            for seg in (system, examples, source, target):
                if len(seg) > 0:
                    if len(result) > 0:
                        seg = add_space(seg, result[-1])
                    elif self.final_lstrip:
                        seg = seg.lstrip()
                    result += (seg,)
            if self.final_rstrip:
                result = result[:-1] + (result[-1].rstrip(),)

            # check prefix caching segments
            if seg_num is not None and seg_num != len(result):
                if conv.is_normalized:
                    result = ("",) * (seg_num - len(result)) + result
                else:
                    raise ValueError(f"seg_num in conversation is not consistent: len({result}) != {seg_num}")

            seg_num = len(result)
            yield result + (conv.num_options,)

    def _to_ppl_prompts(self, conversations: List["Conversation"], split: bool) -> Union[PPLInput, PPLInputSplited]:
        if split:
            results = PPLInputSplited([])
            for *segs, _ in self._get_segs(conversations):
                results.append(tuple(segs))
        else:
            results = PPLInput([])
            for *segs, _ in self._get_segs(conversations):
                results.append(("".join(segs[:-1]), segs[-1]))
        return results

    def _to_prob_prompts(self, conversations: List["Conversation"], split: bool) -> Union[ProbInput, ProbInputSplited]:
        if split:
            results = ProbInputSplited([])
            for segs in self._get_segs(conversations):
                results.append(segs)
        else:
            results = ProbInput([])
            for *segs, o in self._get_segs(conversations):
                results.append(("".join(segs), o))
        return results

    def _to_generation_prompts(
        self,
        conversations: List["Conversation"],
        split: bool,
        max_turns: int = 1,
    ) -> Union[GenInput, GenInputSplited]:
        if split:
            results = GenInputSplited([])
            for *segs, _ in self._get_segs(conversations, max_turns):
                results.append(tuple(segs))
        else:
            results = GenInput([])
            for *segs, _ in self._get_segs(conversations, max_turns):
                results.append("".join(segs))
        return results

    def to_model_prompts(
        self,
        conversations: List["Conversation"],
        split: bool,
        model_evaluation_method: Literal["get_ppl", "get_prob", "generation", "user_defined"],
        max_turns: int = 1,
    ) -> Union[PPLInput, ProbInput, GenInput, PPLInputSplited, ProbInputSplited, GenInputSplited, List["Conversation"]]:
        if model_evaluation_method == "get_ppl":
            return self._to_ppl_prompts(conversations, split)
        elif model_evaluation_method == "get_prob":
            return self._to_prob_prompts(conversations, split)
        elif model_evaluation_method == "generation":
            return self._to_generation_prompts(conversations, split, max_turns)
        else:
            raise ValueError(f"Invalid model_evaluation_method: {model_evaluation_method}")


class Conversation:

    def __init__(
        self,
        messages: Union[str, List[Dict[str, str]], None] = None,
        num_turns: int = 1,
        num_shots: int = 0,
        num_options: int = 1,
        multi_turn_users: Optional[List[str]] = None,
        formatter: Optional[ConversationFormatter] = None,
        model_evaluation_method: Optional[Literal["get_ppl", "get_prob", "generation", "user_defined"]] = None,
        split: Optional[bool] = None,
        is_normalized: bool = False,
    ):
        self.messages = messages if isinstance(messages, list) else []
        self.num_turns = num_turns
        self.num_shots = num_shots
        self.num_options = num_options
        self.mt_users = [] if multi_turn_users is None else multi_turn_users
        self.formatter = formatter
        self.model_evaluation_method = model_evaluation_method
        self.split = split
        self.is_normalized = is_normalized

    @classmethod
    def from_chat(cls, *, user: Optional[str] = None, assistant: Optional[str] = None) -> "Conversation":
        messages = []
        if user:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
        return Conversation(messages=messages)

    @classmethod
    def from_conversations(cls, conversations: List["Conversation"]) -> "Conversation":
        messages = [msg for conv in conversations for msg in conv.messages]
        return Conversation(messages=messages)

    def set_num_options(self, num_options: Optional[int]):
        self.num_options = num_options if isinstance(num_options, int) else 0

    def set_num_shots(self, num_shots: Optional[int]):
        self.num_shots = num_shots if isinstance(num_shots, int) else 0

    def add_multi_turn(self, *, users: Optional[List[str]] = None, assistant: Optional[str] = None):
        if users is not None and assistant is not None:
            raise ValueError("Please provide either users or assistant message.")
        elif users is not None:
            assert len(users) > 1, "At least two users are required for multi-turn conversation."
            self.mt_users = users
            self.num_turns = len(users)
            self.messages.append({"role": "user", "content": self.mt_users.pop(0)})
        elif assistant is not None:
            self.messages.append({"role": "assistant", "content": assistant})
            if len(self.mt_users) > 0:
                self.messages.append({"role": "user", "content": self.mt_users.pop(0)})
        else:
            raise ValueError("Please provide users or assistant message.")

    def get_segs_num(self) -> int:
        if len(self.messages) <= 1:
            return len(self.messages)

        segs = 0
        segs += int(self.messages[0]["role"] == "system")  # system
        segs += 1  # source
        segs += int(self.messages[-1]["role"] == "assistant")  # PPL target
        segs += int(len(self.messages) > segs)  # few-shots
        return segs

    def get_segs(
        self,
        seg: Optional[Literal["system", "examples", "source", "target"]] = None,
    ) -> Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
        """Get splitted segments of the conversation to cache the KV of them."""

        # system
        example_st = 0
        if self.messages[0]["role"] == "system":
            system = [self.messages[0]]
            example_st = 1
        else:
            system = []

        # few-shots example (and previous turns of conversation)
        example_ed = example_st + self.num_shots * 2 + (self.num_turns - len(self.mt_users) - 1) * 2
        examples = self.messages[example_st:example_ed]
        assert all(msg["role"] == "user" for msg in examples[::2]) and all(
            msg["role"] == "assistant" for msg in examples[1::2]
        ), f"Examples should be user and assistant messages, but receive {[msg['role'] for msg in examples]}."

        # target
        source_ed = len(self.messages)
        if self.messages[-1]["role"] == "assistant":
            target = [self.messages[-1]]
            source_ed -= 1
        else:
            target = []

        # source
        source = self.messages[example_ed:source_ed]
        assert source_ed - example_ed <= 1, f"Invalid source messages: {source}"

        results = {"system": system, "examples": examples, "source": source, "target": target}
        if seg:
            return results[seg]
        return results

    def get_generation_results(self) -> Union[str, Tuple[str, ...]]:
        if self.num_turns > 1:
            multi_turn_st = int(self.messages[0]["role"] == "system") + self.num_shots * 2
            # transform to tuples for hashability
            results = tuple(msg["content"] for msg in self.messages[multi_turn_st:] if msg["role"] == "assistant")
            assert len(results) == self.num_turns
            return results
        else:
            assert self.messages[-1]["role"] == "assistant"
            return self.messages[-1]["content"]

    def _merge_system_to_user(self):
        """Whether to convert system message to part of next user message."""
        if self.merge_system_to_user and self.messages[0]["role"] == "system":
            msg = self.messages[0]["content"] + self.system_user_sep + self.messages[1]["content"]

            self.messages.pop(0)
            self.messages[0]["content"] = msg

    def set_formatter(
        self,
        formatter: ConversationFormatter,
        model_evaluation_method: Optional[Literal["get_ppl", "get_prob", "generation", "user_defined"]] = None,
        split: Optional[bool] = None,
    ):
        self.formatter = formatter
        self.model_evaluation_method = model_evaluation_method
        self.split = split and self.get_segs_num() > 1
        self.merge_system_to_user = self.formatter.merge_system_to_user
        self.system_user_sep = self.formatter.system_user_sep
        self._merge_system_to_user()

    def to_model_prompt(
        self,
        max_turns: int = 1,
    ) -> Union[PPLInput, ProbInput, GenInput, PPLInputSplited, ProbInputSplited, GenInputSplited, None]:

        if self.num_turns < max_turns:
            return None
        assert self.formatter is not None, "Please set formatter before converting to model prompt."

        return self.formatter.to_model_prompts(
            [self],
            split=self.split,
            model_evaluation_method=self.model_evaluation_method,
            max_turns=max_turns,
        )[0]

    def apply_prompt_template(self):
        return self.formatter.apply_prompt_template(self)

    def add(
        self,
        other: Optional["Conversation"] = None,
        user: Optional[str] = None,
        assistant: Optional[str] = None,
        inplace: bool = False
    ) -> "Conversation":
        """Add the conversation with another conversation."""
        if inplace:
            return self.add_(other, user, assistant)

        if other is None:
            messages = []
            if user:
                assert isinstance(user, str)
                messages.append({"role": "user", "content": user})
            if assistant:
                assert isinstance(assistant, str)
                messages.append({"role": "assistant", "content": assistant})
        else:
            assert isinstance(other, Conversation)
            messages = other.messages
        conv = Conversation(messages=deepcopy(self.messages))
        # add a copy of other messages
        conv.messages.extend(messages)
        return conv

    def add_(
        self,
        other: Optional["Conversation"] = None,
        user: Optional[str] = None,
        assistant: Optional[str] = None
    ) -> "Conversation":
        """Add the conversation with another conversation inplace."""
        if other is None:
            messages = []
            if user:
                assert isinstance(user, str)
                messages.append({"role": "user", "content": user})
            if assistant:
                assert isinstance(assistant, str)
                messages.append({"role": "assistant", "content": assistant})
        else:
            assert isinstance(other, Conversation)
            messages = other.messages
        # add a copy of other messages
        self.messages.extend(messages)
        return self

    def __iter__(self):
        for message in self.messages:
            yield message

    def __getitem__(self, item):
        return self.messages[item]

    def __setitem__(self, key, value):
        self.messages[key] = value

    def __len__(self):
        return len(self.messages)

    def __repr__(self):
        return "Conversation(\n" + pformat(self.messages) + ")"
