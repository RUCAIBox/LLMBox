OPENAI_COMPLETION_MODELS = ["babbage-002", "davinci-002", "gpt-3.5-turbo-instruct"]
OPENAI_CHAT_MODELS = ["gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-16k", "gpt-4"]
OPENAI_MODELS = OPENAI_COMPLETION_MODELS + OPENAI_CHAT_MODELS
OPENAI_INSTRUCTION_MODELS = ["gpt-3.5-turbo-instruct"] + OPENAI_CHAT_MODELS
ANTHROPIC_MODELS = ["claude-2.1", "claude-instant-1.2"]
DASHSCOPE_MODELS = ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-max-1201", "qwen-max-longcontext",
                    "qwen1.5-72b-chat", "qwen1.5-14b-chat", "qwen1.5-7b-chat", "qwen-72b-chat", "qwen-14b-chat",
                    "qwen-7b-chat", "qwen-1.8b-longcontext-chat", "qwen-1.8b-chat"]
QIANFAN_MODELS = ["ernie-4.0-8k", "ernie-3.5-8k", "ernie-3.5-8k-0205", "ernie-3.5-8k-1222",
                  "ernie-3.5-4k-0205", "ernie-speed-8k", "ernie-speed-128k", "ernie-lite-8k-0922",
                  "ernie-lite-8k-0308", "ernie tiny", "ernie speed-appbuilder"]
