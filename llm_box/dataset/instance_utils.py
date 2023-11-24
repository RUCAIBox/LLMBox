from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from utils import *
from itertools import permutations
import random
import yaml

def knn_construct_examples(instance_query,preformatted_example_data,k):
    """
    select demonstration based on Euclid distance

    Args:
        instance_query (str): target input sequence to be compared
        preformatted_dataset (List[dict]): instance set composed of preformatted instance dic

    Returns:
        List[int]: k nearest examples to the instance_query
    """
    example_data = [preformatted_example_data[i]["data"] for i in range(len(preformatted_example_data))]
    embeddings = []
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    for data in example_data:
        tmp_embeddings = model.encode(data)
        embeddings.append(tmp_embeddings)
    instance_embeddings = model.encode(instance_query)
    embeddings.insert(0,instance_embeddings)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    indice = indices[0][1:]-1
    return indice


def global_entropy_ordering_strategy(indices,labels,preformatted_example_data):
    """
    rank demonstrations based on Global Entropy

    Args:
        indices (List[int]): data indices for permutation
        labels (List[str]): the list of data labels
        preformatted_example_data (List[dict]): instance set composed of preformatted instance dic

    Returns:
        List[int]: best permutation of all instance permutations
    """
    gen_prob_examples = {}
    for perm in permutations(indices):
        gen_prob_examples[perm] = getting_probing_prompt(perm,preformatted_example_data)
    prob_examples = []
    for perm in permutations(indices):
        probe_raw = probe(getting_probing_prompt(perm,preformatted_example_data))
        probe_str = probe_raw.split("choice")[0].strip()
        prob_examples.append(probe_str)
    perm_entropy = {}
    for perm in permutations(indices):
        prompts = []
        for prob_example in prob_examples:
            prompts.append(gen_prob_examples[perm]+prob_example+"\nchoice:")
        outputs = predict_label(prompts,labels)
        eval_results = extract_prediction(outputs,labels)
        label_counts = torch.tensor(eval_results)
        class_distribution = label_counts / label_counts.sum()
        global_entropy = entropy(class_distribution)
        perm_entropy[perm] = global_entropy.float()
    best_perm = max(perm_entropy.keys(), key=lambda k: perm_entropy[k])
    return list(best_perm)

def ape(preformatted_dataset,preformatted_value_dataset):
    """
    generate instructions using APE

    Args:
        preformatted_dataset (List[dict]): preformatted instance set for prompt generation
        preformatted_value_dataset (List[dict]): tpreformatted instance set for prompt evaluation

    Returns:
        List[str]: results of likelihood evaluation
        List[float]: scores based on log probability
    """
    prompt_gen_template = "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n[DEMO]\nThe instruction was to "
    prompt_eval_template = "The instruction was [PROMPT]. Based on the instruction they produced the following input-output pairs:\n\n[DEMO]\n now given the input answer the question:[INPUT].and the answer is:[OUTPUT]"
    with open('/Users/goya/Documents/GitHub/LLMBox/config/ape.yaml') as f:
        config = yaml.safe_load(f)
    # generate prompts
    queries = []
    for i in range(config['generation']['query_num']):
        full_demo = demo_construct(preformatted_dataset,config['generation']['demos_nums'])
        query = prompt_gen_template.replace('[DEMO]',full_demo)
        queries.append(query)
    prompts = generate_instructions(queries,config['generation']['num_prompts_per_query'],config['generation']['gpt_config'])
    # evaluate prompts
    eval_queries = []
    output_indices = []
    for prompt in prompts:
        eval_num = config['evaluation']['eval_num']
        eval_indice = random.sample(range(len(preformatted_value_dataset)), eval_num)
        for i in range(eval_num):
            full_demo = demo_construct(preformatted_dataset,config['evaluation']['demos_nums'])
            eval_query,output_indice = eval_construct(full_demo,preformatted_value_dataset[eval_indice[i]],prompt,prompt_eval_template)
            eval_queries.append(eval_query)
            output_indices.append(output_indice)
    log_probs, predict_tokens = log_probs_(eval_queries, output_indices,config['evaluation']['gpt_config'],config['evaluation']['batch_size'])
    sorted_prompts, sorted_scores = evalute_instructions(prompts,log_probs,config['evaluation']['eval_num'])
    return sorted_prompts,sorted_scores

# unit test
# preformatted_dataset = formatted_copa_data(example_dataset)
# KATE test
# print(knn_construct_examples("this is a test",preformatted_dataset,k))

# APE test
# APE(preformatted_dataset,preformatted_dataset)

# GlobalE test
# print(GlobalEntropyOrderingStrategy(indices,labels,preformatted_dataset))