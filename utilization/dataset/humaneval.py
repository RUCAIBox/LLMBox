import re
from functools import cached_property

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

# Base model may generate a new question and answer it itself. We need to remove the new question or run example.
stop_words = ["def ", "```", "\nif", "\n#"]

def _truncate_code_at_stopwords(code, stop_words):

    # add a newline at the end to facilitate the search
    if not code.endswith("\n"):
        code += "\n"

    # initialize the stop indices
    min_stop_idx = len(code)
    return_statement_idx = 0

    # find the return statement
    match_obj = re.search(r"^\s+return[^\n]*\n", code, re.MULTILINE)
    if match_obj:
        return_statement_idx = match_obj.end()
    else:
        return "# No return statement found\n"

    # find the first stop word after the return statement
    for stop_word in stop_words:
        find_stop_from = -1
        while True:
            stop_index = code.find(stop_word, find_stop_from + 1)
            if stop_index == -1:
                break
            if return_statement_idx <= stop_index:
                min_stop_idx = min(min_stop_idx, stop_index)
                break
            else:
                find_stop_from = stop_index

    code = code[:min_stop_idx]

    # remove repeated "def" statements if any
    def_line = re.search(r"def[^\n]*\n", code)
    if def_line is not None:
        code = code[def_line.end():]

    return code.rstrip()


class Humaneval(GenerationDataset):
    """The dataset of HumanEval.

    The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature, docstring, body, and several unit tests. They were handwritten to ensure not to be included in the training set of code generation models.

    Examples:
        "task_id": "test/0",
        "prompt": "def return1():\n",
        "canonical_solution": "    return 1",
        "test": "def check(candidate):\n    assert candidate() == 1",
        "entry_point": "return1"
    """

    instruction = "{source}"
    example_set = None
    evaluation_set = "test"
    load_args = ("openai_humaneval",)
    extra_model_args = dict(max_tokens=512, temperature=0.1)
    metrics = [PassAtK()]

    def format_instance(self, instance):
        source_text = instance["prompt"].strip()
        target_text = instance["canonical_solution"]
        return dict(source=source_text, target=target_text)

    @staticmethod
    def post_processing(predictions):

        return [_truncate_code_at_stopwords(prediction, stop_words) for prediction in predictions]

    @cached_property
    def references(self):
        return [
            "\n".join(IMPORT_HELPER) + '\n' + instance["prompt"] + "{pred}" + "\n" + instance["test"] + "\n" +
            f"check({instance['entry_point']})" for instance in self.evaluation_data
        ]
