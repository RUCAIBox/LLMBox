import re
from typing import List

from .utils.generation_args import generation_arg

VLLM_ARGS = {
    "temperature": generation_arg(default=0),
    "top_p": generation_arg(),
    "top_k": generation_arg(),
    "max_tokens": generation_arg(default=1024),
    "best_of": generation_arg(needs=lambda b, _: {"use_beam_search": b > 1}),
    "frequency_penalty": generation_arg(),
    "presence_penalty": generation_arg(),
    "repetition_penalty": generation_arg(),
    "length_penalty": generation_arg(),
    "early_stopping": generation_arg(),
    "stop": generation_arg(),
}

HUGGINGFACE_ARGS = {
    "temperature": generation_arg(needs=lambda t, _: {"do_sample": t > 0}),
    "top_p": generation_arg(),
    "top_k": generation_arg(),
    "max_tokens": generation_arg(default=1024, transform_key="max_new_tokens"),
    "best_of": generation_arg(transform_key="num_beams"),
    "repetition_penalty": generation_arg(),
    "length_penalty": generation_arg(),
    "early_stopping": generation_arg(),
    "no_repeat_ngram_size": generation_arg(),
    "stop": generation_arg(),
}

ANTHROPIC_CHAT_COMPLETIONS_ARGS = {
    "max_tokens":
    generation_arg(default=4096),
    "stop":
    generation_arg(
        transform_key="stop_sequences",
        transform_value=lambda x: [i for i in x if not re.match(r"^\s*$", i)],
    ),
    "system":
    generation_arg(),
    "temperature":
    generation_arg(),
    "top_k":
    generation_arg(),
    "top_p":
    generation_arg(),
}

DASHSCOPE_CHAT_COMPLETIONS_ARGS = {
    "temperature": generation_arg(transform_value=lambda x: max(0.0001, x)),
    "top_p": generation_arg(),
    "top_k": generation_arg(),
    "max_tokens": generation_arg(default=1024),
    "repetition_penalty": generation_arg(),
    "enable_search": generation_arg(),
    "stop": generation_arg(),
}

OPENAI_CHAT_COMPLETIONS_ARGS = {
    "frequency_penalty": generation_arg(),
    "logit_bias": generation_arg(),
    "logprobs": generation_arg(),
    "top_logprobs": generation_arg(),
    "max_tokens": generation_arg(default=4096),
    "n": generation_arg(),
    "presence_penalty": generation_arg(),
    "seed": generation_arg(),
    "stop": generation_arg(),
    "temperature": generation_arg(),
    "top_p": generation_arg(),
    "best_of": generation_arg(),
}

OPENAI_COMPLETIONS_ARGS = {
    "best_of": generation_arg(),
    "echo": generation_arg(),
    "frequency_penalty": generation_arg(),
    "logit_bias": generation_arg(),
    "logprobs": generation_arg(),
    "max_tokens": generation_arg(default=1024),
    "n": generation_arg(),
    "presence_penalty": generation_arg(),
    "seed": generation_arg(),
    "stop": generation_arg(),
    "temperature": generation_arg(),
    "top_p": generation_arg(),
}

QIANFAN_CHAT_COMPLETIONS_ARGS = {
    "temperature": generation_arg(transform_value=lambda x: min(max(0.0001, float(x)), 1.0)),
    "top_p": generation_arg(),
    "top_k": generation_arg(),
    "penalty_score": generation_arg(),
    "stop": generation_arg(),
    "disable_search": generation_arg(),
    "enable_citation": generation_arg(),
    "max_tokens":
    generation_arg(default=1024, transform_key="max_output_tokens", transform_value=lambda x: max(2, int(x))),
}


def logit_bias_to_logits_processors(logit_bias: dict) -> List[callable]:

    def logits_processor(logits):
        print(logits.shape)

    return [logits_processor]


VLLM_SERVED_CHAT_COMPLETIONS_ARGS = {
    "n": generation_arg(),
    "best_of": generation_arg(needs={"use_beam_search": True}),
    "presence_penalty": generation_arg(),
    "frequency_penalty": generation_arg(),
    "repetition_penalty": generation_arg(),
    "temperature": generation_arg(),
    "top_p": generation_arg(),
    "top_k": generation_arg(),
    "min_p": generation_arg(),
    "seed": generation_arg(),
    "use_beam_search": generation_arg(),
    "length_penalty": generation_arg(),
    "early_stopping": generation_arg(),
    "stop": generation_arg(),
    "max_tokens": generation_arg(default=1024),
    "logprobs": generation_arg(),
    "echo": generation_arg(extra_body=True),
    "logit_bias": generation_arg(transform_key="logits_processors", transform_value=logit_bias_to_logits_processors),
}

ENDPOINT_ARGS = {
    "dashscope/chat/completions": DASHSCOPE_CHAT_COMPLETIONS_ARGS,
    "anthropic/chat/completions": ANTHROPIC_CHAT_COMPLETIONS_ARGS,
    "openai/chat/completions": OPENAI_CHAT_COMPLETIONS_ARGS,
    "openai/completions": OPENAI_COMPLETIONS_ARGS,
    "qianfan/chat/completions": QIANFAN_CHAT_COMPLETIONS_ARGS,
}

API_MODELS = {
    "babbage-002": {
        "endpoint": "completions",
        "model_type": "base",
        "model_backend": "openai",
    },
    "davinci-002": {
        "endpoint": "completions",
        "model_type": "base",
        "model_backend": "openai",
    },
    "gpt-3.5-turbo-instruct": {
        "endpoint": "completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-3.5-turbo-0125": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-3.5-turbo": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-3.5-turbo-1106": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-3.5-turbo-16k": {
        "endpoint": "completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-3.5-turbo-0613": {
        "endpoint": "completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-3.5-turbo-16k-0613": {
        "endpoint": "completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-4o": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-4-turbo": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-4-turbo-2024-04-09": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-4-turbo-preview": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-4-0125-preview": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-4-1106-preview": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-4-vision-preview": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-4-1106-vision-preview": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-4": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-4-0613": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-4-32k": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "gpt-4-32k-0613": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "openai",
    },
    "claude-3-opus-20240229": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "anthropic",
    },
    "claude-3-sonnet-20240229": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "anthropic",
    },
    "claude-3-haiku-20240307": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "anthropic",
    },
    "claude-2.1": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "anthropic",
    },
    "claude-2.0": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "anthropic",
    },
    "claude-instant-1.2": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "anthropic",
    },
    "qwen-turbo": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen-plus": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen-max": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen-max-0403": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen-max-0107": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen-max-longcontext": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen-max-0428": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen1.5-110b-chat": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen1.5-72b-chat": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen1.5-32b-chat": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen1.5-14b-chat": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen1.5-7b-chat": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen1.5-1.8b-chat": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen1.5-0.5b-chat": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "codeqwen1.5-7b-chat": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen-72b-chat": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen-14b-chat": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen-7b-chat": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen-1.8b-longcontext-chat": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "qwen-1.8b-chat": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "dashscope",
    },
    "ERNIE-3.5-4K-0205": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ERNIE-Speed": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ERNIE-3.5-8K-1222": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ERNIE-3.5-8K-0205": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ERNIE Speed": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ERNIE-Speed-8K": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ERNIE Speed-AppBuilder": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ERNIE 3.5": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ERNIE-Bot-8k": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ERNIE-Lite-8K-0308": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ERNIE-Bot-turbo": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ERNIE-Speed-128k": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ERNIE-Lite-8K-0922": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ERNIE-Bot-4": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ernie-tiny-8k": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ernie-char-8k": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ernie-func-8k": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
    "ai_apaas": {
        "endpoint": "chat/completions",
        "model_type": "chat",
        "model_backend": "qianfan",
    },
}

ERROR_OVERVIEW = {
    "APIConnectionError":
    "Cause: Issue connecting to API model provider's services.\nSolution: Check your network settings, proxy configuration, SSL certificates, or firewall rules.",
    "APITimeoutError":
    "Cause: Request timed out.\nSolution: Retry your request after a brief wait and contact us if the issue persists.",
    "AuthenticationError":
    "Cause: Your API key or token was invalid, expired, or revoked.\nSolution: Check your API key or token and make sure it is correct and active. You may need to generate a new one from your account dashboard.",
    "BadRequestError":
    "Cause: Your request was malformed or missing some required parameters, such as a token or an input.\nSolution: The error message should advise you on the specific error made. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. You may also need to check the encoding, format, or size of your request data.",
    "ConflictError":
    "Cause: The resource was updated by another request.\nSolution: Try to update the resource again and ensure no other requests are trying to update it."
    "InternalServerError"
    "Cause: Issue on API model provider's side.\nSolution: Retry your request after a brief wait and contact us if the issue persists."
    "NotFoundError"
    "Cause: Requested resource does not exist.\nSolution: Ensure you are the correct resource identifier.",
    "PermissionDeniedError":
    "Cause: You don't have access to the requested resource.\nSolution: Ensure you are using the correct API key, organization ID, resource ID and are not accessing from restricted areas.",
    "RateLimitError":
    "Cause: You have hit your assigned rate limit.\nSolution: Pace your requests. Read more in API model provider's Rate limit guide.",
    "UnprocessableEntityError":
    "Cause: Unable to process the request despite the format being correct.\nSolution: Please try the request again.",
}
