from sentence_transformers import SentenceTransformer
from .utils import *
from itertools import permutations
import numpy as np


def knn_construct_examples(instance_query, example_dataset, k):
    """
    select demonstration based on Euclid distance

    Args:
        instance_query (str): target input sequence to be compared
        example_dataset (List[dict]): instance set composed of preformatted instance dic

    Returns:
        List[int]: k nearest examples to the instance_query
    """
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
        data_perm[perm] = getting_prompt(perm, example_dataset)
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
        outputs = predict_label(prompts, labels_num, call_model)
        label_counts = torch.tensor(outputs)
        class_distribution = label_counts / label_counts.sum()
        global_entropy = entropy(class_distribution)
        perm_entropy[perm] = global_entropy.float()
    best_perm = max(perm_entropy.keys(), key=lambda k: perm_entropy[k])
    return list(best_perm)


def ape(example_dataset, eval_dataset, call_model):
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
    # generate prompts
    queries = []
    for i in range(5):
        indice = np.random.choice(len(example_dataset), 5)
        full_demo = getting_prompt(indice, example_dataset)
        query = prompt_gen_template.replace('[DEMO]', full_demo)
        queries.append(query)
    prompts = generate_instructions(queries, 5)
    # evaluate prompts
    eval_queries = []
    for prompt in prompts:
        eval_num = 50
        eval_indice = np.random.choice(len(eval_dataset), eval_num)
        for i in range(eval_num):
            full_demo = getting_prompt(eval_indice, eval_dataset)
            eval_query = eval_construct(full_demo, eval_dataset[eval_indice[i]], prompt, prompt_eval_template)
            eval_queries.append(eval_query)
    ppls = get_ppls(eval_queries, 10, call_model)
    sorted_prompts, sorted_scores = evalute_instructions(prompts, ppls, 50)
    return sorted_prompts[-1]
