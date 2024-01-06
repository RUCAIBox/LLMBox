TRAIN_INPUT=./ptdata/pt_sample_data.txt
LLAMA_TOKENIZER_DIR=/media/public/models/huggingface/llama-7b
MODEL_NAME_OR_PATH=/media/public/models/huggingface/llama-7b
VOCAB_SIZE=3480
OUTPUT_DIR=./output/pt-llama7b

set -e

python spm_merge_tokenizer/merge_tokenizer.py \
    --train_input $TRAIN_INPUT \
    --file_name spm_output \
    --llama_tokenizer_dir $LLAMA_TOKENIZER_DIR \
    --output_sp_dir merged_tokenizer_sp \
    --output_hf_dir merged_tokenizer_hf \
    --vocab_size $VOCAB_SIZE \
    --user_defined_symbols_dir ./ptdata/user_defined_symbols.json

export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 train.py \
    --mode pt \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path $TRAIN_INPUT \
    --tokenizer_name_or_path temp/merged_tokenizer_hf \
    --packing True \
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
