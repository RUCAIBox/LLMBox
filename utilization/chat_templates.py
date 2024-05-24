# sources: https://github.com/huggingface/chat-ui/blob/main/PROMPTS.md

DEFAULT_CHAT_TEMPLATE = (
    "{% macro add(role, msg) -%}"
    "{{ seq[role + '_start'] }}"
    "{{ msg | smart_space(auto_leading_space, seq[role + '_start']) }}"
    "{{ seq[role + '_end'] }}"
    "{%- endmacro %}"
    "{% for message in messages %}"
    "{{ add(message['role'], message['content']) }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ seq['assistant_start'] }}"
    "{% endif %}"
)

DEFAULT_CHAT_CONFIGS = {
    "base": {
        "system_start": "",
        "system_end": "\n\n",
        "user_start": "",
        "user_end": "",
        "assistant_start": "",
        "assistant_end": "\n\n",
        "auto_leading_space": True,
        "default_stops": ["\n"],
    },
    "llama2": {
        "system_start": "<s>[INST] <<SYS>>\n",
        "system_end": "\n<</SYS>>\n\n",
        "user_start": "",
        "user_end": " [/INST] ",
        "assistant_start": "",
        "assistant_end": " </s><s>[INST] ",
        "auto_leading_space": True,
        "default_stops": [""],
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
