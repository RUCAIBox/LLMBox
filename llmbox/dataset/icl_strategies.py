import openai
import torch
from itertools import permutations
import numpy as np
from tqdm import tqdm
from ..model import openai


def knn_construct_examples(instance_query, example_dataset, k):
    """
    select demonstration based on Euclid distance

    Args:
        instance_query (str): target input sequence to be compared
        example_dataset (List[dict]): instance set composed of preformatted instance dic

    Returns:
        List[int]: k nearest examples to the instance_query
    """
    from sentence_transformers import SentenceTransformer

    embeddings = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    model = model.to(device)
    example_data = [example_dataset[i]["source"] for i in range(len(example_dataset))]
    for data in example_data:
        tmp_embeddings = model.encode(data)
        embeddings.append(tmp_embeddings)
    instance_embedding = torch.tensor(model.encode(instance_query))
    distances = [torch.norm(instance_embedding - embedding) for embedding in embeddings]
    indice = torch.topk(torch.tensor(distances), k, largest=False).indices
    return indice


def global_entropy_ordering_strategy(indices, labels, example_dataset, call_model):
    """
    rank demonstrations based on Global Entropy

    Args:
        indices (List[int]): data indices for permutation
        labels (List[int]): the list of data labels
        example_dataset (List[dict]): instance set composed of preformatted instance dic
        call_model: get_ppl function

    Returns:
        List[int]: best permutation of all instance permutations
    """
    data_perm = {}
    # get data permutation demonstration
    for perm in permutations(indices):
        data_perm[perm] = "\n\n".join([example_dataset[i]["source"] + example_dataset[i]["target"] for i in perm])
    # get evalutation indices
    eval_indices = np.random.choice(len(example_dataset), 50)
    perm_entropy = {}
    labels_num = len(labels)
    for perm in permutations(indices):
        prompts = []
        for eval_indice in eval_indices:
            eval_data = example_dataset[eval_indice]
            for j in range(labels_num):
                prompts.append((data_perm[perm] + eval_data["source"], eval_data["options"][j]))
        outputs = [0] * labels_num
        for i in range(len(prompts) // labels_num):
            tmp_prompts = [prompts[labels_num * i + j] for j in range(labels_num)]
            ppls = call_model(tmp_prompts)
            min_ppl_index = ppls.index(min(ppls, key=lambda x: x[0]))
            outputs[min_ppl_index] = outputs[min_ppl_index] + 1
        label_counts = torch.tensor(outputs)
        class_distribution = label_counts / label_counts.sum()
        global_entropy = -(class_distribution * torch.log2(class_distribution)).nansum()
        perm_entropy[perm] = global_entropy.float()
    best_perm = max(perm_entropy.keys(), key=lambda k: perm_entropy[k])
    return list(best_perm)


def ape(example_dataset, eval_dataset, call_model, api_key):
    """
    generate instructions using APE

    Args:
        example_dataset (List[dict]): preformatted instance set for prompt generation
        eval_dataset (List[dict]): preformatted instance set for prompt evaluation

    Returns:
        List[str]: results of likelihood evaluation
        List[float]: scores based on log probability
    """

    class ModelArguments:

        def __init__(self):
            self.model_name_or_path = "gpt-3.5-turbo-instruct"
            self.openai_api_key = api_key
            self.max_tokens = 50
            self.temperature = 0.9

    gpt_config = {
        'n': 5,
        'temperature': 0.9,
        'max_tokens': 50,
        'top_p': 0.9,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0
    }
    prompt_gen_template = "I gave a friend an instruction. Based on the instruction they produced the following sentences:\n\n{DEMO}\nThe instruction was to "
    prompt_eval_template = "The instruction was {PROMPT}. Based on the instruction they produced the following sentences:\n\n{DEMO}\n now evaluate the sentence:{INPUT}"
    # generate prompts
    queries = []
    for i in range(5):
        indice = np.random.choice(len(example_dataset), 5)
        full_demo = "\n\n".join([example_dataset[i]["source"] + example_dataset[i]["target"] for i in indice])
        query = prompt_gen_template.format_map({'DEMO': full_demo})
        queries.append(query)
    prompts = []
    model_parameter = ModelArguments()
    instruct_gen_model = openai.Openai(model_parameter)
    for query in queries:
        response = instruct_gen_model.request(query, gpt_config)
        prompts += [response[i]['text'].strip().replace('\"', '') for i in range(len(response))]

    # evaluate prompts
    eval_queries = []
    demo_indice = np.random.choice(len(eval_dataset), 5)
    eval_indice = np.random.choice(len(eval_dataset), 50)
    for prompt in prompts:
        eval_num = 50
        for i in range(eval_num):
            full_demo = "\n\n".join([eval_dataset[i]["source"] + eval_dataset[i]["target"] for i in demo_indice])
            eval_data = eval_dataset[eval_indice[i]]
            eval_query = prompt_eval_template.format_map({
                'DEMO': full_demo,
                'PROMPT': prompt,
                'INPUT': eval_data["source"]
            }), eval_data["target"]
            eval_queries.append(eval_query)
    ppls = []
    queries_batches = [eval_queries[i:i + 10] for i in range(0, len(eval_queries), 10)]
    print('----------evaluating instructions----------')
    for queries_batch in tqdm(queries_batches):
        batch_ppl = call_model(queries_batch)
        ppls.extend(batch_ppl)

    prompt_avg_log_probs = []
    prompt_num = len(prompts)
    for i in range(prompt_num):
        prompt_avg_log_probs.append([])
        for j in range(50):
            lps = ppls[50 * i + j]
            prompt_avg_log_probs[i].append(lps)

    scores = [np.mean(lps) for lps in prompt_avg_log_probs]
    sorted_prompts = [p for _, p in sorted(zip(scores, prompts))]
    return sorted_prompts[-1]
