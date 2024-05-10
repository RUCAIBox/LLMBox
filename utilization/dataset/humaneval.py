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


class Humaneval(GenerationDataset):
    """The dataset of HumanEval.

    The HumanEval dataset released by OpenAI includes 164 programming problems with a function sig- nature, docstring, body, and several unit tests. They were handwritten to ensure not to be included in the training set of code generation models.

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

    def init_arguments(self):
        self.metrics[0].set_k(k=self.args.pass_at_k)

    def format_instance(self, instance):
        source_text = instance["prompt"].strip()
        target_text = instance["canonical_solution"]
        return dict(source=source_text, target=target_text)

    @staticmethod
    def post_processing(predictions):

        stop_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]

        def _truncate_code_at_stopwords(code, stop_words):
            min_stop_idx = len(code)
            for stop_word in stop_words:
                stop_index = code.find(stop_word)
                if 0 <= stop_index < min_stop_idx:
                    min_stop_idx = stop_index
            return code[:min_stop_idx]

        return [_truncate_code_at_stopwords(prediction, stop_words) for prediction in predictions]

    @property
    def references(self):
        return [
            "\n".join(IMPORT_HELPER) + '\n' + instance["prompt"] + "{pred}" + "\n" + instance["test"] + "\n" +
            f"check({instance['entry_point']})" for instance in self.evaluation_data
        ]
