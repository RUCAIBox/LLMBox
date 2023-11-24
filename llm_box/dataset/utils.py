import time
from logging import getLogger
import random
import yaml
import numpy as np
import openai
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
logger = getLogger(__name__)

def formatted_copa_data(copa_dataset):
    formatted_dataset = []
    for data in copa_dataset:
        item = {}
        formatted_data = data["premise"][:-1]
        if data["question"] == "cause":
            formatted_data += " because \'" + data["choice1"][0].lower() + data["choice1"][1:-1] + '\' or \'' + \
                          data["choice2"][0].lower() + data["choice2"][1:-1] + '\'?'
        elif data["question"] == "effect":
            formatted_data += " therefore \'" + data["choice1"][0].lower() + data["choice1"][1:-1] + '\' or \'' + \
                          data["choice2"][0].lower() + data["choice2"][1:-1] + '\'?'
        item["input"] = formatted_data
        item["output"] = data["label"]
        formatted_dataset.append(item)
    return formatted_dataset

def probe(prompt):
    output = None
    with open('/Users/goya/Documents/GitHub/LLMBox/config/GlobalE.yaml') as f:
        config = yaml.safe_load(f)
    gpt_config = config["gpt_config"]
    while output is None:
        try:
            output = openai.Completion.create(
                **gpt_config,
                prompt=prompt,
            )
        except Exception as e:
            print(e)
            print('Retrying...')
            time.sleep(5)
    return output["choices"][0]["text"]

def getting_probing_prompt(indices,example_data):
    template = "input: {}\nchoice: {}"
    prompt = "\n\n".join(
        [
            template.format(example_data[i]["input"],str(example_data[i]["output"]))
            for i in indices
        ]
        + ["input: "]
    )
    return prompt

def entropy(class_distribution):
    return -(class_distribution * torch.log2(class_distribution)).nansum()

def extract_prediction(outputs,labels):
    labels_length = len(labels)
    labels = [str(labels[i]) for i in range(labels_length)]
    preds = []
    class_dist = [0] * labels_length
    for output in outputs:
        for i, label in enumerate(labels):
            if output["pred_label"][0]==label:
                pred = label
                class_dist[i] += 1
                break
        else:
            raise Exception(
                "predict label does not match any of the labels "
            )
        preds.append(pred)
    return class_dist

def predict_label(prompts,labels):
    def token_to_id(t):
        encoded = tokenizer.encode(t)
        assert len(encoded) == 1
        return encoded[0]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    labels = [str(labels[i]) for i in range(len(labels))]
    label_ids = [token_to_id(labels[i]) for i in range(len(labels))]
    label_id_first = label_ids[0]
    output = []
    for prompt in tqdm(prompts):
        raw = None
        while raw is None:
            try:
                raw = openai.Completion.create(
                    engine="text-curie-001",
                    prompt=prompt,
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=10,
                    logit_bias={str(label_id): 100 for label_id in label_ids},
                )["choices"][0]
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        label_to_logit = {}
        raw_logits = {
            token_to_id(k.strip()): v
            for k, v in raw["logprobs"]["top_logprobs"][0].items()
        }
        assert label_id_first in raw_logits
        for label in raw_logits:
            if label in label_ids and label not in label_to_logit:
                label_to_logit[label] = raw_logits[label]
        probs = torch.tensor([label_to_logit[label] for label in label_ids]).exp()
        probs = probs / probs.sum()
        completion = labels[probs.argmax().item()]
        output.append({"pred_label":completion,"probs":probs})
    return output

def get_token_indices(offsets, log_prob_range):
    lower_index = 0
    for i in range(len(offsets)):
        if offsets[i] <= log_prob_range[0]:
            lower_index = i
        else:
            break

    upper_index = len(offsets)
    for i in range(len(offsets)):
        if offsets[i] >= log_prob_range[1]:
            upper_index = i
            break

    return lower_index, upper_index

def evalute_instructions(prompts,log_probs,eval_num):
    prompt_avg_log_probs = []
    i = 0
    for prompt in prompts:
        prompt_avg_log_probs.append([])
        for _ in range(eval_num):
            lps = log_probs[i]
            prompt_avg_log_probs[-1].append(sum(lps) / len(lps))
            i += 1
    scores = [np.mean(lps) for lps in prompt_avg_log_probs]
    sorted_prompts = [p for _, p in sorted(zip(scores, prompts))]
    sorted_scores = sorted(scores)
    # Reverse both and convert to lists
    sorted_prompts = list(reversed(sorted_prompts))
    sorted_scores = list(reversed(sorted_scores))
    return sorted_prompts,sorted_scores

def log_probs_(queries, output_indices,config,batch_size):
    if not isinstance(queries, list):
        queries = [queries]
    gpt_config = config.copy()
    gpt_config['logprobs'] = 1
    gpt_config['echo'] = True
    gpt_config['max_tokens'] = 0
    queries = [f'\n{queries[i]}' for i in range(len(queries))]

    log_probs = []
    tokens = []
    queries_batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    for queries_batch in tqdm(queries_batches):
        response = None
        while response is None:
            try:
                response = openai.Completion.create(
                    **gpt_config, prompt=queries_batch)
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        log_probs_batch = [response['choices'][i]['logprobs']['token_logprobs'][1:]
                     for i in range(len(response['choices']))]
        tokens_batch = [response['choices'][i]['logprobs']['tokens'][1:]
                  for i in range(len(response['choices']))]
        offsets = [response['choices'][i]['logprobs']['text_offset'][1:]
                   for i in range(len(response['choices']))]
        for i in range(len(tokens_batch)):
            offsets[i] = [offset - 1 for offset in offsets[i]]
            lower_index, upper_index = get_token_indices(
                offsets[i], output_indices[i])
            log_probs_batch[i] = log_probs_batch[i][lower_index:upper_index]
            tokens_batch[i] = tokens_batch[i][lower_index:upper_index]
        log_probs += log_probs_batch
        tokens += tokens_batch
    return log_probs,tokens


def generate_instructions(queries,n,config):
    gpt_config = config.copy()
    gpt_config['n'] = n
    if not isinstance(queries, list):
        queries = [queries]
    result = []
    for query in queries:
        response = None
        while response is None:
            try:
                response = openai.Completion.create(**gpt_config, prompt=query)
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        result += [response['choices'][i]['text'].strip().replace('\"','') for i in range(len(response['choices']))]
    return result

def eval_construct(full_demo,eval_data,prompt,prompt_eval_template):
    eval_input = eval_data["input"]
    eval_output = str(eval_data["output"])
    eval = prompt_eval_template.replace('[DEMO]',full_demo).replace('[PROMPT]',prompt).replace('[INPUT]',eval_input)
    first_idx = eval.find('[OUTPUT]')
    output_indice = first_idx, first_idx + len(eval_output)
    eval = eval.replace('[OUTPUT]', eval_output)
    return eval,output_indice

def demo_construct(dataset,demos_num):
    indices = random.sample(range(len(dataset)), demos_num)
    template = "input: {}\nchoice: {}"
    full_demo = "\n\n".join(
        [
            template.format(dataset[i]["input"], str(dataset[i]["output"]))
            for i in indices
        ]
    )
    return full_demo


def chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

