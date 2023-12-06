from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from .utils import *
from itertools import permutations
import random
import yaml

def knn_construct_examples(instance_query,example_dataset,k):
    """
    select demonstration based on Euclid distance

    Args:
        instance_query (str): target input sequence to be compared
        example_dataset (List[dict]): instance set composed of preformatted instance dic

    Returns:
        List[int]: k nearest examples to the instance_query
    """
    example_data = [example_dataset[i]["source"] for i in range(len(example_dataset))]
    embeddings = []
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    for data in example_data:
        tmp_embeddings = model.encode(data)
        embeddings.append(tmp_embeddings)
    instance_embeddings = model.encode(instance_query)
    embeddings.insert(0,instance_embeddings)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs=-1).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    indice = indices[0][1:]-1
    return indice

def global_entropy_ordering_strategy(indices,labels,example_dataset,call_model):
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
    with open(Path.cwd()/'llm_box'/'dataset'/'config'/'GlobalE.yaml') as f:
        config = yaml.safe_load(f)
    data_perm = {}
    # get data permutation demonstration
    for perm in permutations(indices):
        data_perm[perm] = getting_prompt(perm,example_dataset)
    # get evalutation indices
    eval_indices = np.random.choice(len(example_dataset), config["prob_num"])
    perm_entropy = {}
    labels_num = len(labels)
    for perm in permutations(indices):
        prompts = []
        for eval_indice in eval_indices:
            eval_data = example_dataset[eval_indice]
            for j in range(labels_num):
                prompts.append((data_perm[perm]+eval_data["source"],eval_data["options"][j]))
        outputs = predict_label(prompts,labels_num,call_model)
        label_counts = torch.tensor(outputs)
        class_distribution = label_counts / label_counts.sum()
        global_entropy = entropy(class_distribution)
        perm_entropy[perm] = global_entropy.float()
    best_perm = max(perm_entropy.keys(), key=lambda k: perm_entropy[k])
    return list(best_perm)

def ape(example_dataset,eval_dataset,call_model):
    """
    generate instructions using APE

    Args:
        example_dataset (List[dict]): preformatted instance set for prompt generation
        eval_dataset (List[dict]): preformatted instance set for prompt evaluation

    Returns:
        List[str]: results of likelihood evaluation
        List[float]: scores based on log probability
    """
    prompt_gen_template = "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n[DEMO]\nThe instruction was to "
    prompt_eval_template = "The instruction was [PROMPT]. Based on the instruction they produced the following input-output pairs:\n\n[DEMO]\n now evaluate the sentence:[INPUT]"
    with open(Path.cwd()/'llm_box'/'dataset'/'config'/'ape.yaml') as f:
        config = yaml.safe_load(f)
    # generate prompts
    queries = []
    for i in range(config['generation']['query_num']):
        indice = np.random.choice(len(example_dataset), config['generation']['demos_nums'])
        full_demo = getting_prompt(indice,example_dataset)
        query = prompt_gen_template.replace('[DEMO]',full_demo)
        queries.append(query)
    prompts = generate_instructions(queries,config['generation']['num_prompts_per_query'],config['generation']['gpt_config'])
    # evaluate prompts
    eval_queries = []
    for prompt in prompts:
        eval_num = config['evaluation']['eval_num']
        eval_indice = np.random.choice(len(eval_dataset), eval_num)
        for i in range(eval_num):
            full_demo = getting_prompt(eval_indice,eval_dataset)
            eval_query = eval_construct(full_demo,eval_dataset[eval_indice[i]],prompt,prompt_eval_template)
            eval_queries.append(eval_query)
    ppls = get_ppls(eval_queries,config['evaluation']['batch_size'],call_model)
    sorted_prompts, sorted_scores = evalute_instructions(prompts,ppls,config['evaluation']['eval_num'])
    return sorted_prompts,sorted_scores

# unit test
# preformatted_dataset = formatted_copa_data(example_dataset)
# KATE test
# print(knn_construct_examples("this is a test",preformatted_dataset,k))

# APE test
# APE(preformatted_dataset,preformatted_dataset)

# GlobalE test
# print(GlobalEntropyOrderingStrategy(indices,labels,preformatted_dataset))