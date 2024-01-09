## Supervised Fine-tuning and Continual pretraining using LLMBox

### 1. Continual pretraining with LLama

#### continual pretraining with tokenizer merging

To handle continual pretraining with LLama on new text files, you can add your text files in the ptdata directory and set the correct path in run_7b_pt.sh:

```shell
TRAIN_INPUT=./ptdata/file_name.txt
LLAMA_TOKENIZER_DIR=path/to/llama/tokenizer
MODEL_NAME_OR_PATH=path/to/model/directory
VOCAB_SIZE=the number of tokens
OUTPUT_DIR=path/to/model/output/directory
```

and then execute the following code:

```shell
bash run_7b_pt.sh
```

#### user-defined symbols

If you want to add user-defined symbols when merging new tokenizers, you can rewrite the user_defined_symbols.json. 

```json
{
    "list": [
    	"symbol1",
    	"symbol2"
    ]
}
```

#### tokenizer merging

If you just want to train a tokenizer on your new text files and merge it with llama tokenizer, you can execute the following code:

```bash
TRAIN_INPUT=./ptdata/file_name.txt
LLAMA_TOKENIZER_DIR=path/to/llama/tokenizer
MODEL_NAME_OR_PATH=path/to/model/directory
VOCAB_SIZE=the number of tokens

python spm_merge_tokenizer/merge_tokenizer.py \
    --train_input $TRAIN_INPUT \
    --file_name spm_output \
    --llama_tokenizer_dir $LLAMA_TOKENIZER_DIR \
    --output_sp_dir merged_tokenizer_sp \
    --output_hf_dir merged_tokenizer_hf \
    --vocab_size $VOCAB_SIZE \
    --user_defined_symbols_dir ./data/user_defined_symbols.json
```

#### continual pretraining

If you already have a tokenizer and want to pretrain the model with it, you can set the path to your tokenizer files and directly execute the following code:

```shell
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TRAIN_INPUT=./data/file_name.txt
OUTPUT_DIR=path/to/model/output/directory
MODEL_NAME_OR_PATH=path/to/model/directory
TOKENIZER_NAME_OR_PATH=path/to/tokenizer/directory

torchrun --nproc_per_node=8 train.py \
    --mode pt \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path $TRAIN_INPUT \
    --tokenizer_name_or_path $TOKENIZER_NAME_OR_PATH \
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
```

### 2. Supervised Fine-tuning


