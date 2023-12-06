import time
from logging import getLogger
import random
import yaml
import numpy as np
import openai
import torch
from tqdm import tqdm
logger = getLogger(__name__)

def getting_prompt(indices,example_data):
    prompt = "\n\n".join(
        [
            example_data[i]["source"]+example_data[i]["target"]
            for i in indices
        ]
    )
    return prompt

def entropy(class_distribution):
    return -(class_distribution * torch.log2(class_distribution)).nansum()


def predict_label(prompts,labels_num,call_model):
    output = [0]*labels_num
    for i in range(len(prompts)//labels_num):
        tmp_prompts = [prompts[2*i+j] for j in range(labels_num)]
        ppls = call_model(tmp_prompts)
        min_ppl_index = ppls.index(min(ppls, key=lambda x: x[0]))
        output[min_ppl_index] = output[min_ppl_index]+1
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
    return sorted_prompts,sorted_scores

def get_ppls(queries,batch_size,call_model):
    if not isinstance(queries, list):
        queries = [queries]
    ppls = []
    queries_batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    print('----------evaluating instructions----------')
    for queries_batch in tqdm(queries_batches):
        batch_ppl = call_model(queries_batch)
        ppls.extend(batch_ppl)
    return ppls


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
    eval = prompt_eval_template.replace('[DEMO]',full_demo).replace('[PROMPT]',prompt).replace('[INPUT]',eval_data["source"])
    return (eval,eval_data["target"])


