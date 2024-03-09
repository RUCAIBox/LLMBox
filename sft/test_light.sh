export CUDA_VISIBLE_DEVICES=2,4
export WANDB_MODE=disabled
OUTPUT_DIR=./output/alpaca-7b-lora
torchrun --nproc_per_node=2 train.py \
    --model_name_or_path /media/public/models/huggingface/gpt2 \
    --data_path data/ \
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
    --max_steps 100 \
    --dataset_list chinese.txt \
