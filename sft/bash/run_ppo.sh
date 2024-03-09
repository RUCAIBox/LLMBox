export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=disabled

OUTPUT_DIR=./output/ppo-7b

torchrun --nproc_per_node=8 --master_port=2988 ppo.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --reward_model_name OpenAssistant/reward-model-deberta-v3-large-v2 \
    --save_freq 100 \
    --dataset_name imdb \
    --output_max_length 128 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --ppo_epochs 4 \
    --learning_rate 1.4e-5 \
    --early_stopping True \
    --output_dir $OUTPUT_DIR


