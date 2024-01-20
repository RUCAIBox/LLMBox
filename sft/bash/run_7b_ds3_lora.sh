export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled
OUTPUT_DIR=./output/alpaca-7b-lora
torchrun --nproc_per_node=4 train.py \
    --model_name_or_path /cpfs01/user/GPT/dzc/models/Llama-2-7b-hf \
    --data_path data/alpaca_data_1k.json \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --save_strategy "epoch" \
    --save_steps 2 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --deepspeed configs/ds_z3_bf16.json \
    --lora \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \