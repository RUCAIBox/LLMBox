# This is a legacy test file. For the latest information, please view:
#  - https://github.com/RUCAIBox/LLMBox/blob/main/docs/examples/benchmarking_llama3.sh
#  - https://github.com/RUCAIBox/LLMBox/blob/main/docs/examples/customize_huggingface_model.py
#  - https://github.com/RUCAIBox/LLMBox/blob/main/docs/examples/customize_dataset.py
#  - https://github.com/RUCAIBox/LLMBox/blob/main/docs/utilization/supported-datasets.md

python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d copa --model_type base # 77
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d copa --model_type base --vllm False -b 10 # 76
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d copa --model_type base -shots 64 # 89
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d copa --model_type base -shots 64 --vllm False -b 2 # 87
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d copa --model_type base --vllm False --bnb_config '{"load_in_4bit": true}' # 77
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d copa --model_type base --vllm False --load_in_4bit True # 77
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d copa --model_type base --vllm False --load_in_8bit True # 79
python inference.py -m /data/models/Llama-2-7b-hf-gptq-4bit -d copa --model_type base --gptq True # 78
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d race:middle --evaluation_set test\[:500\] --model_type base # 59.4
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d race:middle --evaluation_set test\[:500\] --model_type base --vllm False -b 4 # 59.4
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d gsm8k --evaluation_set test\[:100\] --model_type base -shots 8 # 16
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d gsm8k --evaluation_set test\[:100\] --model_type base -shots 8 --vllm False -b 4 # 16
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d drop --evaluation_set validation\[:500\] --model_type base --evaluation_set validation\[:500\] # 30 / 17
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d drop --evaluation_set validation\[:500\] --model_type base --evaluation_set validation\[:500\] --vllm False -b 4 # 30 / 17
python inference.py -m /home/tangtianyi/meta-llama/Llama-7b-hf -d winogender:gotcha --model_type base # 59.2, 56.3, 59.2, 58.2
python inference.py -m /home/tangtianyi/meta-llama/Llama-7b-hf -d winogender:gotcha --model_type base # 87.9, 91.9, 90

python inference.py -m /home/tangtianyi/meta-llama/Llama-7b-hf -d mmlu -b 20 --vllm False --model_type base --num_shots 5 # 42.56
python inference.py -m /home/tangtianyi/meta-llama/Llama-7b-hf -d mmlu -b 20 --vllm False --model_type base --num_shots 5 --ranking_type prob # 34.82
python inference.py -m /home/tangtianyi/meta-llama/Llama-7b-hf -d mmlu --model_type base --num_shots 5 --ranking_type prob # 34.88
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d mmlu --model_type base --num_shots 5 --ranking_type prob # 46.5
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-70b-hf -d mmlu --model_type base --num_shots 5 --ranking_type prob # 69.5


python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d bbh --model_type base --num_shots 3 # 33.2
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d bbh --model_type base --num_shots 3 --cot base # 32.6
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d bbh --model_type base --num_shots 3 # 43.5
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d bbh --model_type base --num_shots 3 --cot base # 54.8

python inference.py -m davinci-002 -d copa -b 20 # 88.0
python inference.py -m davinci-002 -d copa -b 20 -shots 1 # 88.0
python inference.py -m davinci-002 -d copa -b 10 -shots 64 # 86.0
python inference.py -m davinci-002 -d race:middle --evaluation_set test\[:500\] -b 20 # 59.4
python inference.py -m davinci-002 -d race:high --evaluation_set test\[:500\] -b 10 # 44.2

python inference.py -m davinci-002 -d openbookqa -b 20 # 59.20
python inference.py -m davinci-002 -d piqa --evaluation_set validation\[:500\] -b 20 # 83.0
python inference.py -m davinci-002 -d winogrande --evaluation_set validation\[:500\] -b 20 # 70.4
python inference.py -m davinci-002 -d winogrande --evaluation_set validation\[:500\] -b 20 -shots 5 # 79.4
python inference.py -m davinci-002 -d arc:ARC-Challenge --evaluation_set test\[:500\] -b 20 # 55.6
python inference.py -m davinci-002 -d arc:ARC-Easy --evaluation_set test\[:500\] -b 20 # 72.0
python inference.py -m davinci-002 -d hellaswag --evaluation_set validation\[:500\] -b 20 -shots 0 # 67.8
python inference.py -m davinci-002 -d hellaswag --evaluation_set validation -b 20 -shots 0 # 79.7
python inference.py -m davinci-002 -d lambada -b 20 --evaluation_set test\[:500\] # 74.4
python inference.py -m davinci-002 -d story_cloze:2016 --dataset_path /home/tangtianyi/data/story_cloze --evaluation_set test\[:500\] -b 20 # 83.4
python inference.py -m davinci-002 -d crows_pairs --evaluation_set test\[:500\] -b 20 # 66.6

python inference.py -m davinci-002 -d nq --evaluation_set validation\[:500\] -b 20 # 33 / 21
python inference.py -m davinci-002 -d nq --evaluation_set validation\[:500\] -b 20 -shots 1 # 38 / 26
python inference.py -m davinci-002 -d triviaqa --evaluation_set validation\[:500\] -b 20 # 70 / 63
python inference.py -m davinci-002 -d webq --evaluation_set test\[:500\] -b 20 # 37 / 19

python inference.py -m davinci-002 -d real_toxicity_prompts --evaluation_set train\[:100\] -b 20 --proxy_port 1428 --perspective_api_key AIzaSyA8au5E8NZ5-RLMgYEwCgo5-rCT6-1FjD4 # 6.39

python inference.py -m gpt-3.5-turbo-instruct -d tldr --evaluation_set test\[:500\] -b 20 # 22.02
python inference.py -m gpt-3.5-turbo-instruct -d cnn_dailymail --evaluation_set test\[:500\] -b 20 # 21.79
python inference.py -m gpt-3.5-turbo-instruct -d squad_v2 --evaluation_set validation\[:500\] -b 20 # 55 / 47
python inference.py -m gpt-3.5-turbo-instruct -d squad_v2 --evaluation_set validation\[:500\] -b 20 -shots 4 # 59 / 53
python inference.py -m gpt-3.5-turbo-instruct -d quac --evaluation_set validation\[:500\] -b 10 # 43 / 16
python inference.py -m gpt-3.5-turbo-instruct -d drop --evaluation_set validation\[:500\] -b 20 -shots 4 # 73 / 64
python inference.py -m davinci-002 -d drop --evaluation_set validation\[:500\] -b 20 # 42 / 24
# only 500 instances
# python inference.py -m davinci-002 -d coqa -b 20 # 65 / 48

python inference.py -m gpt-3.5-turbo -d math --evaluation_set test\[:100\] -b 20 # 42
python inference.py -m gpt-3.5-turbo -d math --evaluation_set test\[:100\] -b 20 -shots 4 # 52
python inference.py -m gpt-3.5-turbo -d gsm8k --evaluation_set test\[:100\] -b 20 # 66
python inference.py -m gpt-3.5-turbo -d gsm8k --evaluation_set test\[:100\] -b 20 -shots 8 # 72
python inference.py -m gpt-3.5-turbo -d gsm8k --evaluation_set test\[:100\] -b 20 -shots 8 --cot least_to_most # 72
python inference.py -m gpt-3.5-turbo -d gsm8k --evaluation_set test\[:100\] -b 20 -shots 8 --cot pal # 77
python inference.py -m gpt-3.5-turbo -d gsm8k --evaluation_set test\[:100\] -b 20 -shots 8 --temperature 0.7 --sample_num 5 # 76
python inference.py -m gpt-3.5-turbo -d mt_bench # 8.62



python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d gsm8k --model_type base --num_shots 8 # 14.6 ok
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d math --model_type base --num_shots 4 # 1.5
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d piqa --model_type base # 78.8 ok
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d siqa --model_type base # 45.6
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d hellaswag --model_type base # 75.0
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d winogrande --model_type base # 69.4 ok
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d openbookqa --model_type base # 54.2
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d arc:ARC-Challenge --model_type base # 45.6
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d arc:ARC-Easy --model_type base # 69.7
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d commonsenseqa --model_type base # 45.7
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d nq --model_type base -shots 5 # 35.7/26.7 ok
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d triviaqa --model_type base -shots 5 # 74.8/70.1 
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d quac --model_type base # 21.8/6.0
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d boolq --model_type base # 78.4 ok
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d squad --model_type base # 59.2/43.5
python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d squad --model_type base -shots 5 # 86.1/76.4

python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d gsm8k --model_type base --num_shots 8 # 56.6 ok
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d math --model_type base --num_shots 4 # 6.04
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d piqa --model_type base # 82.1
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d siqa --model_type base # 46.6
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d hellaswag --model_type base # 83.4
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d winogrande --model_type base # 80.4 ok
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d openbookqa --model_type base # 53.6
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d arc:ARC-Challenge --model_type base # 60.1 ok
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d arc:ARC-Easy --model_type base # 76.4
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d commonsenseqa --model_type base # 47.9
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d nq --model_type base -shots 5 # 50.0/39.6 ok
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d triviaqa --model_type base -shots 5 # 89.4/85.2, /87.6
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d quac --model_type base # 27.5/11.4
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d boolq --model_type base # 85.6 ok
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d squad --model_type base # 70.6/55.9
python inference.py -m /mnt/tangtianyi/Llama-2-70b-hf -d squad --model_type base -shots 5 # 90.4/82.0
