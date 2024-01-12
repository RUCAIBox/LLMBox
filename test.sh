python inference.py -m davinci-002 -d copa -b 20 # 88.0
python inference.py -m davinci-002 -d copa -b 20 -shots 1 # 87.0
python inference.py -m davinci-002 -d copa -b 20 -shots 64 # 89.0
python inference.py -m davinci-002 -d race:middle --evaluation_set test\[:500\] -b 20 # 59.4
python inference.py -m davinci-002 -d race:high --evaluation_set test\[:500\] -b 20 # 44.2

python inference.py -m gpt-3.5-turbo-instruct -d tldr --evaluation_set test\[:500\] -b 20 # 21.01
python inference.py -m gpt-3.5-turbo-instruct -d cnn_dailymail --evaluation_set test\[:500\] -b 20 # 18.23
python inference.py -m gpt-3.5-turbo-instruct -d squad_v2 --evaluation_set validation\[:500\] -b 20 # 53 / 43
python inference.py -m gpt-3.5-turbo-instruct -d squad_v2 --evaluation_set validation\[:500\] -b 20 -shots 4 # 54 / 47
python inference.py -m gpt-3.5-turbo-instruct -d quac --evaluation_set validation\[:500\] -b 20 # 39 / 14
python inference.py -m gpt-3.5-turbo-instruct -d drop  --evaluation_set validation\[:500\] -b 20 -shots 4 # 69 / 59
python inference.py -m davinci-002 -d drop --evaluation_set validation\[:500\] -b 20 # 18 / 10
# only 500 instances
python inference.py -m davinci-002 -d coqa -b 20 # 65 / 48

python inference.py -m gpt-3.5-turbo -d gsm8k --evaluation_set test\[:100\] -b 20 # 78
python inference.py -m gpt-3.5-turbo -d gsm8k --evaluation_set test\[:100\] -b 20 -shots 8 # 73
python inference.py -m gpt-3.5-turbo -d math --evaluation_set test\[:100\] -b 20 # 37
python inference.py -m gpt-3.5-turbo -d math --evaluation_set test\[:100\] -b 20 -shots 4 # 44

python inference.py -m gpt-3.5-turbo -d gsm8k --evaluation_set test\[:10\] -b 20 # 50
python inference.py -m gpt-3.5-turbo -d gsm8k --evaluation_set test\[:10\] -b 20 -shots 8 # 70
python inference.py -m gpt-3.5-turbo -d gsm8k --evaluation_set test\[:10\] -b 20 -shots 8 --cot least_to_most # 60
python inference.py -m gpt-3.5-turbo -d gsm8k --evaluation_set test\[:10\] -b 20 -shots 8 --cot pal # 60
python inference.py -m gpt-3.5-turbo -d gsm8k --evaluation_set test\[:10\] -b 20 -shots 8 --temperature 0.7 --sample_num 5 # 80