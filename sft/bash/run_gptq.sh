export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR=./output/gptq_model
python gptq.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --output_dir $OUTPUT_DIR \
    --bits 4 \
    --group_size 128 \
    --desc_act True \
    --damp_percent 0.1 \
    --num_samples 128 \
    --seq_len 512 \
    --use_triton False \
    --batch_size 1 \
    --cache_examples_on_gpu True \
    --use_fast True \
    --trust_remote_code False \
    --unquantized_model_dtype float16
