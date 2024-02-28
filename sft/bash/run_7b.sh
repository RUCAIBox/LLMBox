export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=2,4
OUTPUT_DIR=output/dolly
torchrun --nproc_per_node=2 --master_port=2977 train.py \
    --model_name_or_path /home/tangtianyi/Llama-7b-hf \
    --data_path /home/luowenyang/data/dolly.jsonl \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 1 \
    --save_total_limit 100 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --deepspeed configs/ds_z3_bf16.json \
    --gradient_checkpointing True \
    --tf32 True \
    --packing False \