python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d copa --model_type base # 77
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d copa --model_type base --vllm False -b 10 # 76
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d copa --model_type base -shots 64 # 89
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d copa --model_type base -shots 64 --vllm False -b 2 # 87
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d race:middle --evaluation_set test\[:500\] --model_type base # 59.4
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d race:middle --evaluation_set test\[:500\] --model_type base --vllm False -b 4 # 59.4
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d gsm8k --evaluation_set test\[:100\] --model_type base -shots 8 # 16
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d gsm8k --evaluation_set test\[:100\] --model_type base -shots 8 --vllm False -b 4 # 16
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d drop --evaluation_set validation\[:500\] --model_type base --evaluation_set validation\[:500\] # 30 / 17
python inference.py -m /home/tangtianyi/meta-llama/Llama-2-7b-hf -d drop --evaluation_set validation\[:500\] --model_type base --evaluation_set validation\[:500\] --vllm False -b 4 # 30 / 17

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

python inference.py -m davinci-002 -d nq --evaluation_set validation\[:500\] -b 20 # 33 / 21
python inference.py -m davinci-002 -d nq --evaluation_set validation\[:500\] -b 20 -shots 1 # 38 / 26
python inference.py -m davinci-002 -d triviaqa --evaluation_set validation\[:500\] -b 20 # 70 / 63
python inference.py -m davinci-002 -d webq --evaluation_set test\[:500\] -b 20 # 37 / 19

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
