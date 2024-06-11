from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from logging import getLogger
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from ...model import openai_model
from ...utils.arguments import ModelArguments

logger = getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from ...model.model import ApiModel, Model


@dataclass
class ICLUtilMixin:

    ape: bool
    globale: bool
    kate: bool

    def set_icl(self, kate: bool, globale: bool, ape: bool, model: "Model"):
        self.ape = ape
        self.globale = globale
        self.kate = kate

        if self.ape:
            self.set_ape(model)
        if self.globale:
            self._set_globale(model)
        if self.kate:
            self._set_kate()

    def _set_get_ppl(self, model: "Model"):
        try:
            model.get_ppl([])
        except NotImplementedError:
            raise NotImplementedError("GlobalE requires a model with a get_ppl method")
        except Exception:
            pass

        self._get_ppl = model.get_ppl

    def set_ape(self, model: "Model"):

        import openai

        self._set_get_ppl(model)
        model_args = ModelArguments(
            model_name_or_path="gpt-3.5-turbo-instruct",
            openai_api_key=openai.api_key,
            max_tokens=50,
            temperature=0.9,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        self._instruct_gen_model = openai_model.Openai(model_args)

    def _set_globale(self, model: "Model"):
        self._set_get_ppl(model)

    def _set_kate(self):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        paraphrase_model_name = "paraphrase-MiniLM-L6-v2"
        self._paraphrase_model = SentenceTransformer(paraphrase_model_name, device=device)
        self._paraphrase_model.eval()
        logger.info(f"kate model {paraphrase_model_name} loaded: {self._paraphrase_model}")

        self._embeddings = []

    def generate_ape(self, example_dataset: List[Dict[str, str]], eval_dataset: Iterable[Dict[str, str]]):
        return generate_ape(self._instruct_gen_model, example_dataset, list(eval_dataset), self._get_ppl)

    def global_entropy_ordering_strategy(
        self, indices: List[int], labels: List[int], example_dataset: List[Dict[str, str]]
    ):
        return global_entropy_ordering_strategy(indices, labels, example_dataset, self._get_ppl)

    def knn_construct_examples(
        self, instance_query: str, example_dataset: List[Dict[str, str]], k: int, batch_size: int = 32
    ):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        paraphrase_encode = lambda x, show_progress_bar=False: self._paraphrase_model.encode(
            x,
            convert_to_tensor=True,
            convert_to_numpy=False,
            show_progress_bar=show_progress_bar,
            device=device,
            batch_size=batch_size,
        )

        if len(self._embeddings) == 0:
            self._embeddings = paraphrase_encode([example["source"] for example in example_dataset], True)

        return knn_construct_examples(paraphrase_encode, instance_query, self._embeddings, k)


def knn_construct_examples(
    paraphrase_encode: callable,
    instance_query: str,
    example_embeddings: List[torch.Tensor],
    k: int,
):
    """
    select demonstration based on Euclid distance

    Args:
        instance_query (str): target input sequence to be compared
        example_dataset (List[dict]): instance set composed of preformatted instance dic

    Returns:
        List[int]: k nearest examples to the instance_query
    """

    instance_embedding = paraphrase_encode(instance_query)
    distances = [torch.norm(instance_embedding - embedding) for embedding in example_embeddings]
    indice = torch.topk(torch.tensor(distances), k, largest=False).indices
    return indice


def global_entropy_ordering_strategy(indices, labels, example_dataset, get_ppl):
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
            ppls = get_ppl(tmp_prompts)
            min_ppl_index = ppls.index(min(ppls, key=lambda x: x[0]))
            outputs[min_ppl_index] = outputs[min_ppl_index] + 1
        label_counts = torch.tensor(outputs)
        class_distribution = label_counts / label_counts.sum()
        global_entropy = -(class_distribution * torch.log2(class_distribution)).nansum()
        perm_entropy[perm] = global_entropy.float()
    best_perm = max(perm_entropy.keys(), key=lambda k: perm_entropy[k])
    return list(best_perm)


def generate_ape(instruct_gen_model: "ApiModel", example_dataset, eval_dataset, get_ppl):
    """
    generate instructions using APE

    Args:
        example_dataset (List[dict]): preformatted instance set for prompt generation
        eval_dataset (List[dict]): preformatted instance set for prompt evaluation

    Returns:
        List[str]: results of likelihood evaluation
        List[float]: scores based on log probability
    """

    prompt_gen_template = "I gave a friend an instruction. Based on the instruction they produced the following sentences:\n\n{DEMO}\nThe instruction was to "
    prompt_eval_template = "The instruction was {PROMPT}. Based on the instruction they produced the following sentences:\n\n{DEMO}\n now evaluate the sentence:{INPUT}"
    # generate prompts
    queries = []
    for i in range(5):
        indice = np.random.choice(len(example_dataset), 5)
        full_demo = "\n\n".join([example_dataset[i]["source"] + example_dataset[i]["target"] for i in indice])
        query = prompt_gen_template.format_map({"DEMO": full_demo})
        queries.append(query)
    prompts = []
    for query in queries:
        response = instruct_gen_model.request(query, n=5)
        prompts += [response[i]["text"].strip().replace('"', "") for i in range(len(response))]

    # evaluate prompts
    eval_queries = []
    demo_indice = np.random.choice(len(eval_dataset), 5)
    eval_indice = np.random.choice(len(eval_dataset), 50)
    for prompt in prompts:
        eval_num = 50
        for i in range(eval_num):
            full_demo = "\n\n".join([eval_dataset[i]["source"] + eval_dataset[i]["target"] for i in demo_indice])
            eval_data = eval_dataset[eval_indice[i]]
            eval_query = (
                prompt_eval_template.format_map({
                    "DEMO": full_demo,
                    "PROMPT": prompt,
                    "INPUT": eval_data["source"]
                }),
                eval_data["target"],
            )
            eval_queries.append(eval_query)
    ppls = []
    queries_batches = [eval_queries[i:i + 10] for i in range(0, len(eval_queries), 10)]
    logger.info("APE: evaluating instructions")
    for queries_batch in tqdm(queries_batches):
        batch_ppl = get_ppl(queries_batch)
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
