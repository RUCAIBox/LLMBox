export CUDA_VISIBLE_DEVICES=7,8
export WANDB_MODE=disabled
OUTPUT_DIR=./output/alpaca-7b
torchrun train.py \
    --model_name_or_path /media/public/models/huggingface/gpt2 \
    --data_path data/alpaca_data_1k.json \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 8 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --deepspeed configs/ds_z3_bf16.json \
    --tf32 True \
    --gradient_checkpointing True \
