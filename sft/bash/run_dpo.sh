export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=disabled
OUTPUT_DIR=./output/dpo_model

torchrun --nproc_per_node=2 --master_port=2988 dpo.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path data/dpo_exp_data.json \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --deepspeed configs/ds_z3_bf16.json \
    --tf32 True \
    --gradient_checkpointing True

