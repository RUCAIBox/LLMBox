import json
import os
import random
import re
import string
import argparse

import numpy as np
import tqdm
from rouge_score import rouge_scorer
from gensim.summarization import bm25

from llm_box.model.openai import Openai
from llm_box.utils import ModelArguments

# tokenizer for chinese instruction
from transformers import AutoTokenizer

checkpoint = "bigscience/bloomz-7b1"
tokenizer_cn = AutoTokenizer.from_pretrained(checkpoint)
os.environ['TOKENIZERS_PARALLELISM'] = 'True'

prompt_en = '''You are asked to come up with a set of 20 diverse task instructions. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.\n\
Here are the requirements:\n\
1. Try not to repeat the verb for each instruction to maximize diversity.\n\
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instrucitons.\n\
3. The type of instructions should be diverse. The list should include diverse types of tasks like open-ended generation, classification, editing, etc.\n\
2. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.\n\
3. The instructions should be in English.\n\
4. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.\n\
5. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.\n\
6. Not all instructions require input. For example, when a instruction asks about some general information, "what is the highest peak in the world", it is not necssary to provide a specific context. In this case, we simply put "<noinput>" in the input field.\n\
7. The output should be an appropriate response to the instruction and the input. Make sure the output is less than 100 words.\n\
List of 20 tasks:\n'''

prompt_cn = '''你被要求提供10个多样化的任务指令。这些任务指令将被提供给GPT模型，我们将评估GPT模型完成指令的能力。\n\
以下是你提供指令需要满足的要求：\n\
1.尽量不要在每个指令中重复动词，要最大化指令的多样性。\n\
2.使用指令的语气也应该多样化。例如，将问题与祈使句结合起来。\n\
3.指令类型应该是多样化的，包括各种类型的任务，类别种类例如：brainstorming，open QA，closed QA，rewrite，extract，generation，classification，chat，summarization。\n\
4.GPT语言模型应该能够完成这些指令。例如，不要要求助手创建任何视觉或音频输出。例如，不要要求助手在下午5点叫醒你或设置提醒，因为它无法执行任何操作。例如，指令不应该和音频、视频、图片、链接相关，因为GPT模型无法执行这个操作。\n\
5.指令用中文书写，指令应该是1到2个句子，允许使用祈使句或问句。\n\
6.你应该给指令生成适当的输入，输入字段应包含为指令提供的具体示例，它应该涉及现实数据，不应包含简单的占位符。输入应提供充实的内容，使指令具有挑战性。\n\
7.并非所有指令都需要输入。例如，当指令询问一些常识信息，比如“世界上最高的山峰是什么”，不需要提供具体的上下文。在这种情况下，我们只需在输入字段中放置“<无输入>”。当输入需要提供一些文本素材（例如文章，文章链接）时，就在输入部分直接提供一些样例。当输入需要提供音频、图片、视频或者链接时，则不是满足要求的指令。\n\
8.输出应该是针对指令和输入的恰当回答。 \n\
下面是10个任务指令的列表：\n'''


def encode_prompt(prompt_instructions, language):
    """Encode multiple prompt instructions into a single string."""
    # prompt template
    prompt = prompt_en if language == 'en' else prompt_cn
    noinput_text = "<noinput>" if language == 'en' else "<无输入>"
    instruction_text = "Instruction" if language == 'en' else "指令"
    input_text = "Input" if language == 'en' else "输入"
    output_text = "Output" if language == 'en' else "输出"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")

        input = noinput_text if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. {instruction_text}: {instruction}\n"
        prompt += f"{idx + 1}. {input_text}:\n{input}\n"
        prompt += f"{idx + 1}. {output_text}:\n{output}\n"

    prompt += f"###\n"
    prompt += f"{idx + 2}. {instruction_text}:"
    return prompt


# post-process
def post_process_gpt3_response(num_prompt_instructions, response, language):
    '''Standardize and filter responses'''
    if response is None:
        return []
    try:  #for gpt-3.5-turbo
        raw_instructions = response["message"]["content"]
        print("gpt-3.5-turbo")
    except:
        try:
            raw_instructions = response["text"]  #for gpt-3.5-turbo-instruct
        except:
            print("ERROR parse!")

    # standardize instance like(Instruction: Input: Output:)
    if language == 'en':
        if 'Instruction:' not in raw_instructions[0:10]:
            raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + raw_instructions
    elif language == 'cn':
        if '指令:' not in raw_instructions[0:10] and '指令：' not in raw_instructions[0:10]:
            raw_instructions = f"{num_prompt_instructions+1}. 指令:" + raw_instructions

    raw_instructions = re.split("###", raw_instructions)
    instructions = []

    # process_gpt3_response_en
    if language == 'en':
        for idx, inst in enumerate(raw_instructions):
            # if the decoding stops due to length, the last example is likely truncated so we discard it
            if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
                continue
            idx += num_prompt_instructions + 1
            splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
            if len(splitted_data) != 7:
                continue
            else:
                inst = splitted_data[2].strip()
                input = splitted_data[4].strip()
                input = "" if input.lower() == "<noinput>" else input
                output = splitted_data[6].strip()
            # 1. filter out too short or too long instructions
            if len(inst.split()) <= 3 or len(inst.split()) > 150:
                continue
            # 2. filter based on keywords that are not suitable for language models.
            blacklist = [
                "image",
                "images",
                "graph",
                "graphs",
                "picture",
                "pictures",
                "file",
                "files",
                "map",
                "maps",
                "draw",
                "plot",
                "go to",
                "video",
                "audio",
                "music",
                "flowchart",
                "diagram",
            ]
            blacklist += []
            if any(find_word_in_string(word, inst, language) for word in blacklist):
                continue

            # 3. Note this is not a comprehensive filtering for all programming instructions.
            if inst.startswith("Write a program"):
                continue
            # 4. filter those starting with punctuation
            if inst[0] in string.punctuation:
                continue
            # 5. filter those starting with non-english character
            if not inst[0].isascii():
                continue
            instructions.append({"instruction": inst, "input": input, "output": output})

    # process_gpt3_response_cn
    elif language == 'cn':
        blacklist = [
            "图像", "图片", "照片", "文件", "图表", "图层", "曲线图", "折线图", "直线图", "柱形图", "饼状图", "链接", "http", 'OpenAI', 'chatgpt',
            'gpt-3', 'gpt-3.5', 'gpt-4'
        ]
        replace_empty_list = [
            '要求GPT模型能够', '要求GPT能够', '要求GPT模型', '让GPT模型', '使用GPT模型', '请向GPT模型', 'GPT模型应', 'GPT模型应该', '请求GPT模型',
            '需要GPT模型回答', '请GPT模型', '请让GPT模型', '训练GPT模型', 'GPT模型需要', '要求GPT', '让GPT', '使用GPT', '请向GPT', 'GPT应', 'GPT应该',
            '请求GPT', '需要GPT回答', '请GPT', '请让GPT', '训练GPT', 'GPT需要', '希望GPT模型能够', '希望GPT能够', '以便GPT模型能够', '以便GPT能够',
            '使得GPT模型能够', '使得GPT能够', '使GPT模型能够', '使GPT能够', '由GPT模型', '使GPT模型'
        ]
        for idx, inst in enumerate(raw_instructions):
            # if the decoding stops due to length, the last example is likely truncated so we discard it
            if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
                continue
            # filter based on keywords that are not suitable for language models.
            if any(find_word_in_string(word, inst, language) for word in blacklist):
                continue

            # extract instance (指令，输入，输出)
            intruction_pattern = re.compile(
                r"(?<=(?:" + '|'.join(['指令:', '指令：']) + "))[\s\S]*?(?=" + '|'.join(['输入:', '输入：']) + ")"
            )
            input_pattern = re.compile(
                r"(?<=(?:" + '|'.join(['输入:', '输入：']) + "))[\s\S]*?(?=" + '|'.join(['输出:', '输出：']) + ")"
            )
            output_pattern = re.compile(r"(?<=(?:" + '|'.join(['输出:', '输出：']) + "))[\s\S]*?(?=$)")
            intruction_match = intruction_pattern.search(inst)
            input_match = input_pattern.search(inst)
            output_match = output_pattern.search(inst)

            if intruction_match and input_match and output_match:
                inst = re.sub(r'\d+\.$', '', intruction_match.group().strip()).strip('\n')
                input = re.sub(r'\d+\.$', '', input_match.group().strip()).strip('\n')
                input = "" if "无输入" in input else input
                output = output_match.group().strip().strip('\n')
                if '指令:' in output and '输入:' in output and '输出:' in output:  # Return the first entry if not separated by '###'
                    output_pattern_new = re.compile(r"(?<=(?:" + "))[\s\S]*?(?=" + '|'.join(['指令:', '指令：']) + ")")
                    output_match_new = output_pattern_new.search(output)
                    if output_match_new:
                        output = re.sub(r'\d+\.$', '', output_match_new.group().strip()).strip('\n')

                # filter unreasonable instructions
                # 1. filter out too short inst
                if len(inst) <= 3:
                    continue
                # 2. filter out inst with 'GPT'
                for item in replace_empty_list:
                    inst = inst.replace(item, "")
                if "GPT" in inst or 'GPT' in input:
                    continue
                if len(input) == 0:  # No input
                    instructions.append({"instruction": inst, "input": input, "output": output})
                else:
                    if '示例' in inst or '例子' in inst:  # Provide examples in inst
                        if len(inst) < 150:
                            instructions.append({"instruction": inst, "input": input, "output": output})
                    else:  # No examples given
                        if len(inst) < 100:
                            instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s, language):
    if language == "en":
        return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)
    elif language == "cn":
        return w in s
    else:
        raise ValueError("Unsupported language: {0}".format(language))


def load_tasks_file(language, output_dir, seed_tasks_path):
    '''load seed instructions, LM-generated instructions and merge them'''
    # load the seed instructions
    if not seed_tasks_path:
        seed_tasks_path = "./seed_tasks.jsonl" if language == 'en' else "./zh_seed_tasks.json"

    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [{
        "instruction": t["instruction"],
        "input": t["instances"][0]["input"],
        "output": t["instances"][0]["output"]
    } for t in seed_tasks]

    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")
    os.makedirs(output_dir, exist_ok=True)

    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        generated_file_path = os.path.join(output_dir, "regen.json")
        with open(generated_file_path, 'r') as file:
            machine_instruction_data = json.load(file)
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # merge all instructions
    all_instructions = [d["instruction"]
                        for d in seed_instruction_data] + [d["instruction"] for d in machine_instruction_data]

    return (seed_instruction_data, machine_instruction_data, all_instructions)


def call_chatgpt(prompt, language):
    model_args = ModelArguments(model_name_or_path='gpt-3.5-turbo-instruct')
    openai_instance = Openai(model_args)
    model_args_for_request = {
        'temperature': 1,
        'max_tokens': 3072
        if language == 'en' else 1024,  # hard-code to maximize the length. the requests will be automatically adjusted
        'top_p': 1.0,
        'frequency_penalty': 0,
        'presence_penalty': 0,
        'stop': ["\n20", "20.", "20."],
        'logit_bias': {
            "50256": -100
        },  # prevent the <|endoftext|> token from being generated
        'n': 1,
        'presence_penalty': 0.0,
        'frequency_penalty': 0.0,
    }
    response = openai_instance.request(prompt, model_args_for_request)
    return response


def generate_instruction_following_data(
    language='en',
    output_dir="./",
    seed_tasks_path="",
    num_instructions_to_generate=100,
    num_prompt_instructions=3,
):
    # load seed instructions, LM-generated instructions and merge them
    (seed_instruction_data, machine_instruction_data,
     all_instructions) = load_tasks_file(language, output_dir, seed_tasks_path)

    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)

    # first we tokenize all the seed instructions
    if language == 'en':
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]
    elif language == 'cn':
        all_instruction_tokens = [tokenizer_cn.tokenize(inst) for inst in all_instructions]
        bm25Model = bm25.BM25(all_instruction_tokens)

    # generated new machine instructions
    request_idx = 0
    initial_instruction_num = len(machine_instruction_data)

    while len(machine_instruction_data) < num_instructions_to_generate + initial_instruction_num:
        request_idx += 1

        # construct prompt text
        # only sampling from the seed tasks
        prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
        prompt = encode_prompt(prompt_instructions, language)
        # call chatgpt to generate instance
        response = call_chatgpt(prompt, language)
        # post process
        instruction_data = post_process_gpt3_response(num_prompt_instructions, response[0], language)
        # filter based on similarity
        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            if language == 'en':
                new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
                rouge_scores = []
                for tokens in all_instruction_tokens:
                    score = rouge_scorer._score_lcs(new_instruction_tokens, tokens)
                    rouge_scores.append(score.fmeasure)
            elif language == 'cn':
                new_instruction_tokens = tokenizer_cn.tokenize(instruction_data_entry["instruction"])
                rouge_scores = bm25Model.get_scores(new_instruction_tokens)

            # set a threshold
            threshold_score = 0.7 if language == 'en' else 18
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i]
                for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > threshold_score:
                continue
            else:
                keep += 1

            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))

            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)

    print(f"Generated {len(machine_instruction_data) - initial_instruction_num} new machine-generated instructions")
    # save the regenerated instructions
    output_file_path = os.path.join(output_dir, "regen.json")
    with open(output_file_path, mode='w', encoding="utf-8") as output_file:
        json.dump(
            machine_instruction_data, output_file, indent=4, default=str, ensure_ascii=False
        )  # non-ASCII characters can appear directly in JSON


def main(**kwargs):
    parser = argparse.ArgumentParser(description="Generate instruction-following data.")

    # Add command-line arguments
    parser.add_argument('--language', default='en', help='Language for data generation')
    parser.add_argument('--output_dir', default='./', help='Output directory for generated data')
    parser.add_argument('--seed_tasks_path', default='', help='Path to seed tasks file')
    parser.add_argument(
        '--num_instructions_to_generate', type=int, default=100, help='Number of instructions to generate'
    )
    parser.add_argument('--num_prompt_instructions', type=int, default=3, help='Number of prompt instructions')

    args = parser.parse_args()
    args_dict = vars(args)

    # Pass the arguments using dictionary unpacking
    generate_instruction_following_data(**args_dict)


if __name__ == "__main__":
    main()
