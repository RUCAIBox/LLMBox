export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
OUTPUT_DIR=./output/ppo_model

accelerate launch --num_machines 1  --num_processes 1 \
    ppo.py \
    --model_name /home/textbox/align/trl/Llama-2-7b-hf \
    --reward_model_name OpenAssistant/reward-model-deberta-v3-large-v2 \
    --adafactor False \
    --save_freq 100 \
    --dataset_name imdb \
    --output_max_length 128 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --batched_gen True \
    --ppo_epochs 4 \
    --learning_rate 1.4e-5 \
    --early_stopping True \
    --output_dir $OUTPUT_DIR