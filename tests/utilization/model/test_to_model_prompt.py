import pytest

from utilization.model.model_utils.conversation import Conversation, ConversationFormatter

from ..fixtures import conversation

model_evaluation_methods = {
    ("generation", False): (
        "This is a system message.\n"
        "\n"
        "This is a user message. This is an assistant message.\n"
        "\n"
        "This is the second user message."
    ),
    ("generation", True): (
        "This is a system message.\n\n",
        "This is a user message. This is an assistant message.\n\n",
        "This is the second user message.",
    ),
    ("get_ppl", False): (
        "This is a system message.\n"
        "\n"
        "This is a user message. This is an assistant message.\n"
        "\n"
        "This is the second user message.",
        " This is the second assistant message.",
    ),
    ("get_ppl", True): (
        "This is a system message.\n\n",
        "This is a user message. This is an assistant message.\n\n",
        "This is the second user message.",
        " This is the second assistant message.",
    ),
    ("get_prob", False): (
        "This is a system message.\n"
        "\n"
        "This is a user message. This is an assistant message.\n"
        "\n"
        "This is the second user message. This is the second assistant message.", 1
    ),
    ("get_prob", True): (
        "This is a system message.\n\n",
        "This is a user message. This is an assistant message.\n\n",
        "This is the second user message.",
        " This is the second assistant message.",
        1,
    ),
}


@pytest.mark.parametrize("model_evaluation_method, split", model_evaluation_methods.keys())
def test_to_model_prompt(conversation: Conversation, model_evaluation_method: str, split: bool):
    formatter = ConversationFormatter.from_chat_template("base")
    conversation.set_formatter(formatter, model_evaluation_method, split)
    conversation.set_num_shots(1)  # 1 shot
    conversation.set_num_options(1)
    if model_evaluation_method == "generation":
        conversation.messages = conversation.messages[:-1]

    expected = model_evaluation_methods[(model_evaluation_method, split)]
    assert conversation.to_model_prompt() == expected
