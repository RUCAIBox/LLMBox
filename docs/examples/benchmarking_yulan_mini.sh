export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

MODEL=/home/huyiwen/miniyulan-ckpts/miniyulan-2B-s25d-decay80-26_20241125_004748/checkpoint-246000-rebalanced
LOG_LEVEL=debug

source activate base

# Math500
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d math500 -shots 4 --max_example_tokens 3300 --vllm True --stop "\n\nQuestion" "\n\Solve" "\n\nAnswer" --cot k0_math --max_tokens 796

# HumanEval (Pass@1)
# python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d humaneval --vllm True -shots 0 --max_example_tokens 2560 --pass_at_k 1 --sample_num 20 --temperature 0.1

# GSM8K (CoT & PAL)
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d gsm8k -b 80:auto --vllm False -shots 7 --max_example_tokens 3500 --max_new_tokens 596 --stop "\n\nQuestion" "\n\nAnswer" --cot k0_math
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d gsm8k -b 150:auto --vllm False -shots 5 --max_example_tokens 3500 --max_new_tokens 596 --cot pal

# LAMBADA
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d lambada --vllm True -shots 0 --max_example_tokens 3072

# MMLU (ppl_no_option & prob)
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d mmlu -b 256:auto -shots 5 --max_example_tokens 3072 --vllm False --ranking_type ppl_no_option
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d mmlu -b 256:auto -shots 5 --max_example_tokens 3072 --vllm False --ranking_type prob

# CMMLU (ppl_no_option & prob)
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d cmmlu -b 128:auto -shots 5 --max_example_tokens 2560 --ranking_type ppl_no_option
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d cmmlu -b 128:auto -shots 5 --max_example_tokens 2560 --ranking_type prob

# MBPP
# python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d mbpp --vllm False -b 120:auto -shots 3 --max_example_tokens 3072 --max_new_tokens 1024 --pass_at_k 1 --sample_num 20 --temperature 0.1 --stop '\n[DONE]'

# Story Cloze
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d story_cloze -b 512:auto -shots 0 --max_example_tokens 3072 --prefix_caching True --dataset_path '/home/huyiwen/story_cloze'

# TLDR
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d tldr --vllm True -shots 0 --max_example_tokens 3072 --prefix_caching True --max_new_tokens 1024

# HellaSwag
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d hellaswag -b 512:auto -shots 5 --max_example_tokens 2560

# ARC Easy & Challenge (ppl_no_option & prob)
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d arc -b 64:auto -shots 0 --max_example_tokens 2560 --vllm False --ranking_type ppl_no_option
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d arc -b 64:auto -shots 0 --max_example_tokens 2560 --vllm False --ranking_type prob

# QuAC
inference quac --vllm False -shots 0 --max_example_tokens 3072 --prefix_caching True

# RACE
inference race -b 100:auto -shots 0 --max_example_tokens 3072 --prefix_caching True

# TriviaQA
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d triviaqa --vllm True -shots 5 --max_example_tokens 2560

# IMBUE Private
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d imbue_private -b 256:auto -shots 0 --max_example_tokens 3072 --prefix_caching True

# BoolQ (ppl_no_option & prob)
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d boolq -b 64 -shots 0 --ranking_type prob
python inference.py -m $MODEL --hf_mirror --prefix_caching True --log_level $LOG_LEVEL -d boolq -b 64 -shots 0 --ranking_type ppl_no_option

# IMBUE Code
inference imbue_code -b 512:auto -shots 0 --max_example_tokens 3072 --prefix_caching True

