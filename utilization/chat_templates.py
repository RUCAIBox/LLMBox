from typing import Any, Dict, List, Union

__all__ = ["DEFAULT_CHAT_TEMPLATE", "DEFAULT_CHAT_CONFIGS"]


def smart_space(parts: List[str], auto_leading_space) -> str:

    def add_space(msg: str, auto_leading_space: bool, context: str) -> str:
        if auto_leading_space and msg and context and not context[-1].isspace():
            return ' ' + msg
        return msg

    rendered = ""
    for part in parts:
        rendered += add_space(part, auto_leading_space, rendered)
    return rendered


# sources: https://github.com/huggingface/chat-ui/blob/main/PROMPTS.md

DEFAULT_CHAT_TEMPLATE = (
    "{%- set data = namespace(parts=[]) -%}"
    ""
    "{%- if 'all_start' in seq -%}"
    "{%- set data.parts = data.parts + [seq['all_start']] -%}"
    "{%- endif -%}"
    ""
    "{%- for message in messages -%}"
    "{%- set data.parts = data.parts + [seq[message['role'] + '_start']] -%}"
    "{%- set data.parts = data.parts + [message['content']] -%}"
    "{%- set data.parts = data.parts + [seq[message['role'] + '_end']] -%}"
    "{%- endfor -%}"
    ""
    "{%- if add_generation_prompt -%}"
    "{%- set data.parts = data.parts + [seq['assistant_start']] -%}"
    "{%- endif -%}"
    ""
    "{{ data.parts | smart_space(auto_leading_space) }}"
)

# Chat configs format:
#
# A jinja2 chat-template accessing to variables:
#   - messages: A list of dictionaries with the following keys:
#   - seq: A dictionary with the `starts` and `ends` sequences of each role.
#   - smart_space: A filter that controls the leading space of a string.
#   - add_generation_prompt: Whether to prepend assistant_start after entire chat message.
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
#   - default_stops: A list of strings that indicate the end of a message.
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
        "default_stops": [],
    },
    "llama2": {
        "all_start": "<s>[INST] ",
        "system_start": "<<SYS>>\n",
        "system_end": "\n<</SYS>>\n\n",
        "user_start": "",
        "user_end": " [/INST] ",
        "assistant_start": "",
        "assistant_end": " </s><s>[INST] ",
        "auto_leading_space": True,
        "default_stops": [],
    },
    "chatml": {
        "system_start": "<|im_start|>system\n",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n",
        "auto_leading_space": True,
        "default_stops": ["<|im_end|>"],
    },
    "zephyr": {
        "system_start": "<|system|>\n",
        "system_end": "</s>\n",
        "user_start": "<|user|>\n",
        "user_end": "</s>\n",
        "assistant_start": "<|assistant|>\n",
        "assistant_end": "</s>\n",
        "auto_leading_space": True,
        "default_stops": ["</s>"],
    },
    "phi3": {
        "system_start": "<|system|>\n",
        "system_end": "<|end|>\n",
        "user_start": "<|user|>\n",
        "user_end": "<|end|>\n",
        "assistant_start": "<|assistant|>\n",
        "assistant_end": "<|end|>\n",
        "auto_leading_space": True,
        "default_stops": ["<|end|>"],
    },
    "llama3": {
        "system_start": "<|start_header_id|>system<|end_header_id|>\n\n",
        "system_end": "<|eot_id|>",
        "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_end": "<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant_end": "<|eot_id|>",
        "auto_leading_space": True,
        "default_stops": ["<|eot_id|>"],
    },
    "alpaca": {
        "system_start": "### Input:\n",
        "system_end": "\n\n",
        "user_start": "### Instruction:\n",
        "user_end": "\n\n",
        "assistant_start": "### Response:\n",
        "assistant_end": "\n\n",
        "auto_leading_space": True,
        "default_stops": ["###"],
    }
}
