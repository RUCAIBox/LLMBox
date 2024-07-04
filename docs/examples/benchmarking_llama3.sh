export CUDA_VISIBLE_DEVICES=0

# Load model from huggingface hub
BASE_MODEL=meta-llama/Meta-Llama-3-8B
INSTRUCT_MODEL=meta-llama/Meta-Llama-3-8B-Instruct

############ BASE MODEL ############

# MMLU (5-shot)
python inference.py \
  --model $BASE_MODEL \
  --batch_size 128:auto \
  --dataset mmlu \
  --num_shots 5 \
  --ranking_type prob \
  --max_example_tokens 4096

# AGIEval English (5-shot)
python inference.py \
  --model $BASE_MODEL \
  --dataset agieval:[English] \
  --num_shots 5 \
  --batch_size 16:auto \
  --max_example_tokens 2560

# CommonSenseQA (7-shot)
python inference.py \
  --model $BASE_MODEL \
  --batch_size 128:auto \
  --dataset commonsenseqa \
  --num_shots 7

# Winogrande (5-shot)
python inference.py \
  --model $BASE_MODEL \
  --batch_size 128:auto \
  --dataset winogrande \
  --num_shots 5

# Big-Bench Hard (3-shot)
python inference.py \
  --model $BASE_MODEL \
  --dataset bbh \
  --cot base \
  --num_shots 3 \
  --vllm

# ARC-Challenge (25-shot)
python inference.py \
  --model $BASE_MODEL \
  --batch_size 32:auto \
  --dataset arc:ARC-Challenge \
  --num_shots 25 \
  --ranking_type prob \
  --max_example_tokens 4096

# TriviaQA-Wiki (5-shot)
python inference.py \
  --model $BASE_MODEL \
  --dataset triviaqa \
  --num_shots 5

# SQuAD (1-shot, EM) and QuAC (1-shot, F1)
python inference.py \
  --model $BASE_MODEL \
  --dataset squad_v2 quac \
  --num_shots 1 \
  --max_example_tokens 4096

# BoolQ (0-shot)
python inference.py \
  --model $BASE_MODEL \
  --batch_size 128:auto \
  --dataset boolq

# DROP (3-shot, F1)
python inference.py \
  --model $BASE_MODEL \
  --dataset drop \
  --num_shots 3


############ INSTRUCT MODEL ############

# MMLU (5-shot)
python inference.py \
  --model $INSTRUCT_MODEL \
  --batch_size 128:auto \
  --dataset mmlu \
  --max_example_tokens 4096 \
  --ranking_type prob \
  --num_shots 5

# GPQA (0-shot)
python inference.py \
  --model $INSTRUCT_MODEL \
  --dataset gpqa \
  --ranking_type prob

# HumanEval (0-shot)
python inference.py \
  --model $INSTRUCT_MODEL \
  --dataset humaneval \
  --pass_at_k 1 \
  --sample_num 20 \
  --temperature 0.1 \
  --max_example_tokens 2560

# GSM-8K (8-shot, CoT)
python inference.py \
  --model $INSTRUCT_MODEL \
  --dataset gsm8k \
  --num_shots 8 \
  --max_example_tokens 2560 \
  --cot pal

# MATH (4-shot, CoT)
python inference.py \
  --model $INSTRUCT_MODEL \
  --dataset math \
  --num_shots 4 \
  --max_example_tokens 2560 \
  --cot base
