TRAINING_DATA=./data/chinese.txt
VOCAB_SIZE=4000
OUTPUT_DIR=./output/llama-chinese

python merge_tokenizer.py \
    --input $TRAINING_DATA \
    --vocab_size $VOCAB_SIZE \
    --output_dir $OUTPUT_DIR \
    --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
    --user_defined_symbols_dir data/user_defined_symbols.json

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=disabled
torchrun --nproc_per_node=8 train.py \
    --mode pt \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path $TRAINING_DATA \
    --tokenizer_name_or_path $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --save_strategy "epoch" \
    --save_steps 2 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --deepspeed configs/ds_z3_bf16.json \
