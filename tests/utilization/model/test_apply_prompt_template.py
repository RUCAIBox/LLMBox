from utilization.chat_templates import DEFAULT_CHAT_TEMPLATE
from utilization.model.model_utils.conversation import Conversation, ConversationFormatter

from ..fixtures import conversation


def test_base(conversation: Conversation):
    formatter = ConversationFormatter.from_chat_template("base")
    conversation.set_formatter(formatter)
    formatted_conversation = conversation.apply_prompt_template()
    assert formatted_conversation == (
        "This is a system message.\n"
        "\n"
        "This is a user message. This is an assistant message.\n"
        "\n"
        "This is the second user message. This is the second assistant message."
    )


def test_llama2(conversation: Conversation):
    formatter = ConversationFormatter.from_chat_template("llama2")
    conversation.set_formatter(formatter)
    formatted_conversation = conversation.apply_prompt_template()
    assert formatted_conversation == (
        "<s>[INST] <<SYS>>\n"
        "This is a system message.\n"
        "<</SYS>>\n"
        "\n"
        "This is a user message. [/INST] This is an assistant message. </s><s>[INST] This is the second user message. [/INST] This is the second assistant message. </s><s>[INST] "
    )


def test_no_smart_space(conversation: Conversation):
    prompt_config = {
        "system_start": "",
        "system_end": "",
        "user_start": "",
        "user_end": "",
        "assistant_start": "",
        "assistant_end": "",
        "auto_leading_space": False,
        "default_stops": [],
    }
    formatter = ConversationFormatter(prompt_config, DEFAULT_CHAT_TEMPLATE)
    conversation.set_formatter(formatter)
    formatted_conversation = conversation.apply_prompt_template()
    assert formatted_conversation == (
        "This is a system message.This is a user message.This is an assistant message.This is the second user message.This is the second assistant message."
    )


def test_smart_space(conversation: Conversation):
    prompt_config = {
        "system_start": "",
        "system_end": "",
        "user_start": "",
        "user_end": "",
        "assistant_start": "",
        "assistant_end": "",
        "auto_leading_space": True,
        "default_stops": [],
    }
    formatter = ConversationFormatter(prompt_config, DEFAULT_CHAT_TEMPLATE)
    conversation[2]["content"] = " This is an assistant message."  # extra leading space
    conversation.set_formatter(formatter)
    formatted_conversation = conversation.apply_prompt_template()
    assert formatted_conversation == (
        "This is a system message. This is a user message. This is an assistant message. This is the second user message. This is the second assistant message."
    )


def test_final_strip(conversation: Conversation):
    prompt_config = {
        "system_start": "\n\n",
        "system_end": "",
        "user_start": "",
        "user_end": "",
        "assistant_start": "",
        "assistant_end": "\n\n",
        "auto_leading_space": True,
        "final_lstrip": False,
        "final_rstrip": False,
        "default_stops": [],
    }
    formatter = ConversationFormatter(prompt_config, DEFAULT_CHAT_TEMPLATE)
    conversation.set_formatter(formatter)
    formatted_conversation = conversation.apply_prompt_template()
    assert formatted_conversation == (
        "\n\nThis is a system message. This is a user message. This is an assistant message.\n\nThis is the second user message. This is the second assistant message.\n\n"
    )
