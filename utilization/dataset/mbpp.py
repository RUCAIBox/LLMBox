import re

from ..metric import PassAtK
from .generation_dataset import GenerationDataset

IMPORT_HELPER = [
    "import math",
    "import re",
    "import sys",
    "import copy",
    "import datetime",
    "import itertools",
    "import collections",
    "import heapq",
    "import functools",
    "import hashlib",
    "import numpy",
    "import numpy as np",
    "import string",
    "from typing import *",
    "from collections import *",
]


class Mbpp(GenerationDataset):
    """The dataset of MBPP.

    The benchmark consists of around 1,000 crowd-sourced Python programming problems, designed to be solvable by entry level programmers, covering programming fundamentals, standard library functionality, and so on. Each problem consists of a task description, code solution and 3 automated test cases.

    Examples:
        task_id: 601
        text: Write a function to find the longest chain which can be formed from the given set of pairs.
        code: class Pair(object): def __init__(self, a, b): self.a = a self.b = b def max_chain_length(arr, n): max = 0 mcl = [1 for i in range(n)] for i in range(1, n): for j in range(0, i): if (arr[i].a > arr[j].b and mcl[i] < mcl[j] + 1): mcl[i] = mcl[j] + 1 for i in range(n): if (max < mcl[i]): max = mcl[i] return max
        test_list: [ "assert max_chain_length([Pair(5, 24), Pair(15, 25),Pair(27, 40), Pair(50, 60)], 4) == 3", "assert max_chain_length([Pair(1, 2), Pair(3, 4),Pair(5, 6), Pair(7, 8)], 4) == 4", "assert max_chain_length([Pair(19, 10), Pair(11, 12),Pair(13, 14), Pair(15, 16), Pair(31, 54)], 5) == 5" ]
        test_setup_code:
        challenge_test_list: []
    """

    instruction = ""
    example_set = "train"
    evaluation_set = "test"
    load_args = ("mbpp", "full")
    extra_model_args = dict(stop=['\n[DONE]'], temperature=0.1)
    metrics = [PassAtK()]

    def __init__(self, dataset_name, args, model, subset_name=None):
        super().__init__(dataset_name, args, model, subset_name=subset_name)
        self.metrics[0].set_k(k=args.pass_at_k)

    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        super().load_raw_dataset(dataset_path, subset_name, evaluation_set, example_set)
        self.example_data = EXAMPLARS

    def format_instance(self, instance):
        prompt = instance["text"]
        tests = '\n'.join(instance["test_list"])
        code = instance["code"].replace("\r", "").replace("\t", "    ")
        source_prompt = f"You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{tests}\n"
        target_prompt = f"[BEGIN]\n{code}\n[DONE]\n"
        return dict(source=source_prompt, target=target_prompt)

    def post_processing(self, predictions):
        # answer_pattern = re.compile(r"\[BEGIN\](.*?)\[DONE\]", re.DOTALL)
        # return [
        #     re.search(answer_pattern, p).group(1).strip() if re.search(answer_pattern, p) else p for p in predictions
        # ]
        return [p.split("[DONE]")[0].split("[BEGIN]\n")[-1] for p in predictions]

    @property
    def references(self):
        return [
            "\n".join(IMPORT_HELPER) + '\n' + "{pred}" + "\n" + "\n".join(instance["test_list"])
            for instance in self.evaluation_data
        ]


EXAMPLARS = [{
    "text":
    "Write a function to find the similar elements from the given two tuple lists.",
    "code":
    "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
    "task_id":
    2,
    "test_setup_code":
    "",
    "test_list": [
        "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
        "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
        "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"
    ],
    "challenge_test_list": []
}, {
    "text":
    "Write a python function to identify non-prime numbers.",
    "code":
    "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
    "task_id":
    3,
    "test_setup_code":
    "",
    "test_list":
    ["assert is_not_prime(2) == False", "assert is_not_prime(10) == True", "assert is_not_prime(35) == True"],
    "challenge_test_list": []
}, {
    "text":
    "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
    "code":
    "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
    "task_id":
    4,
    "test_setup_code":
    "",
    "test_list": [
        "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
        "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
        "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"
    ],
    "challenge_test_list": []
}]
