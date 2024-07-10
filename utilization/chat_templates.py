from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = ["DEFAULT_CHAT_TEMPLATE", "DEFAULT_CHAT_CONFIGS", "add_space", "smart_space"]


def add_space(
    msg: str,
    context: str,
    auto_leading_space: bool = True,
    remove_space_between: bool = True,
    starts: Optional[List[str]] = None,
    ends: Optional[List[str]] = None
) -> str:
    if starts is None or ends is None or remove_space_between is False:
        context_ends_special = False
        msg_starts_special = False
    else:
        context_ends_special = any(context.endswith(e) for e in ends if len(e) > 0)
        msg_starts_special = any(msg.startswith(s) for s in starts if len(s) > 0)
    if (auto_leading_space and msg and context)\
            and not (context[-1].isspace() or msg[0].isspace())\
            and not (context_ends_special and msg_starts_special):
        return ' ' + msg
    return msg


def smart_space(
    parts: List[Tuple[str, bool]], auto_leading_space: bool, remove_space_between: bool, seq: List[str]
) -> str:
    starts = [seq[role + "_start"] for role in ["system", "user", "assistant"] if (role + "_start") in seq]
    ends = [seq[role + "_end"] for role in ["system", "user", "assistant"] if (role + "_end") in seq]
    if "bos_token" in seq:
        ends.append(seq["bos_token"])
    rendered = ""
    for part in parts:
        if part[0]:
            rendered += add_space(
                part[0],
                rendered,
                auto_leading_space=auto_leading_space and part[1],
                remove_space_between=remove_space_between,
                starts=starts,
                ends=ends
            )
    return rendered


# sources: https://github.com/huggingface/chat-ui/blob/main/PROMPTS.md

DEFAULT_CHAT_TEMPLATE = (
    "{%- set data = namespace(parts=[]) -%}"
    ""
    "{%- if 'all_start' in seq -%}"
    "{%- set data.parts = data.parts + [(seq['all_start'], False)] -%}"
    "{%- endif -%}"
    ""
    "{%- for message in messages -%}"
    "{%- set data.parts = data.parts + [(seq[message['role'] + '_start'], False)] -%}"
    "{%- set data.parts = data.parts + [(message['content'], True)] -%}"
    "{%- set data.parts = data.parts + [(seq[message['role'] + '_end'], False)] -%}"
    "{%- endfor -%}"
    ""
    "{%- if add_gen_prompt -%}"
    "{%- set data.parts = data.parts + [(seq['assistant_start'], False)] -%}"
    "{%- set data.parts = data.parts + [(seq['generation_prompt'], True)] -%}"
    "{%- endif -%}"
    ""
    "{{ data.parts | smart_space(auto_leading_space, remove_space_between, seq) }}"
)

# Chat configs format:
#
# A jinja2 chat-template accessing to variables:
#   - messages: A list of dictionaries with the following keys:
#   - seq: A dictionary with the `starts` and `ends` sequences of each role.
#   - smart_space: A filter that controls the leading space of a string.
#   - add_gen_prompt: Whether to prepend assistant_start after entire chat message.
#
# or a dictionary with the following keys:
#   - all_start: The string to prepend to the entire chat message.
#   - system_start: The string to prepend to the system message.
#   - system_end: The string to append to the system message.
#   - user_start: The string to prepend to the user message.
#   - user_end: The string to append to the user message.
#   - assistant_start: The string to prepend to the assistant message.
#   - assistant_end: The string to append to the assistant message.
#   - auto_leading_space: Whether to add a leading space when concatenating two
#     strings if the first string does not end with a whitespace.
#   - default_stop: A list of strings that indicate the end of a message.
#
DEFAULT_CHAT_CONFIGS: Dict[str, Union[Dict[str, Any], str]] = {
    "base": {
        "system_start": "",
        "system_end": "\n\n",
        "user_start": "",
        "user_end": "",
        "assistant_start": "",
        "assistant_end": "\n\n",
        "auto_leading_space": True,
        "final_rstrip": True,
        "remove_space_between": False,
        "default_stop": [],
    },
    "llama2": {
        "system_start": "<<SYS>>\n",
        "system_end": "\n<</SYS>>\n\n",
        "user_start": "<s>[INST] ",
        "user_end": " [/INST] ",
        "assistant_start": "",
        "assistant_end": " </s>",
        "auto_leading_space": True,
        "final_rstrip": False,
        "remove_space_between": True,
        "default_stop": [],
    },
    "chatml": {
        "system_start": "<|im_start|>system\n",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n",
        "auto_leading_space": True,
        "final_rstrip": False,
        "remove_space_between": True,
        "default_stop": ["<|im_end|>"],
    },
    "zephyr": {
        "system_start": "<|system|>\n",
        "system_end": "</s>\n",
        "user_start": "<|user|>\n",
        "user_end": "</s>\n",
        "assistant_start": "<|assistant|>\n",
        "assistant_end": "</s>\n",
        "auto_leading_space": True,
        "final_rstrip": False,
        "remove_space_between": True,
        "default_stop": ["</s>"],
    },
    "phi3": {
        "system_start": "<|system|>\n",
        "system_end": "<|end|>\n",
        "user_start": "<|user|>\n",
        "user_end": "<|end|>\n",
        "assistant_start": "<|assistant|>\n",
        "assistant_end": "<|end|>\n",
        "auto_leading_space": True,
        "final_rstrip": False,
        "remove_space_between": True,
        "default_stop": ["<|end|>", "<|endoftext|>"],
    },
    "llama3": {
        "system_start": "<|start_header_id|>system<|end_header_id|>\n\n",
        "system_end": "<|eot_id|>",
        "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_end": "<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant_end": "<|eot_id|>",
        "auto_leading_space": True,
        "final_rstrip": False,
        "remove_space_between": True,
        "default_stop": ["<|eot_id|>"],
    },
    "alpaca": {
        "system_start": "### Input:\n",
        "system_end": "\n\n",
        "user_start": "### Instruction:\n",
        "user_end": "\n\n",
        "assistant_start": "### Response:\n",
        "assistant_end": "\n\n",
        "auto_leading_space": True,
        "final_rstrip": False,
        "remove_space_between": False,
        "default_stop": ["###"],
    }
}
