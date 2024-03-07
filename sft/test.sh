export WANDB_MODE=disabled
OUTPUT_DIR=./output/alpaca-7b
python train.py \
    --mode hb \
    --max_steps 100 \
    --model_name_or_path gpt2 \
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
    --tf32 False \
    --no_cuda True \
    --packing False \
    --dataset_list chinese.txt,alpaca_data_1k.json \
    --dataset_ratio [0.7,0.3]
