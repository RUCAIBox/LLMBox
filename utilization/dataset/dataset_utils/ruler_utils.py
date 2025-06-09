# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import os
import random
import re
import uuid
from functools import lru_cache, cache
from typing import List, Union, Literal
import datasets

import numpy as np
from packaging.version import parse as parse_version
from importlib.metadata import version

from tqdm import tqdm


def generate_samples(
    haystack,
    TOKENIZER=None,
    *,
    max_seq_length: int,
    type_haystack: str,
    type_needle_k: str,
    type_needle_v: str,
    template: str,
    num_samples: int = 500,
    tokens_to_generate: int = 128,
    num_needle_v: int = 1,
    num_needle_k: int = 1,
    num_needle_q=1,
    incremental: int = 500,
    remove_newline_tab: bool = False,
    random_seed: int = 42,
) -> list[dict]:
    assert TOKENIZER is not None, "TOKENIZER is not defined."
    num_needle_k = max(num_needle_k, num_needle_q)
    write_jsons = []
    tokens_to_generate = tokens_to_generate

    if type_haystack == "essay":
        incremental = 500
    elif type_haystack == "repeat":
        incremental = 25
    elif type_haystack == "needle":
        incremental = 25

    if type_haystack != "essay" and max_seq_length < 4096:
        incremental = 5

    num_haystack = incremental

    total_tokens = 0  # Track the total tokens generated for the first example
    while total_tokens + tokens_to_generate < max_seq_length:
        input_text, answer, query = generate_input_output(
            num_haystack,
            haystack,
            type_haystack=type_haystack,
            num_needle_k=num_needle_k,
            type_needle_k=type_needle_k,
            num_needle_v=num_needle_v,
            type_needle_v=type_needle_v,
            template=template,
            num_needle_q=num_needle_q,
            random_seed=random_seed,
        )
        # Calculate the number of tokens in the example
        total_tokens = len(TOKENIZER(input_text + " ".join(answer)).input_ids)
        if total_tokens + tokens_to_generate > max_seq_length:
            num_haystack -= incremental
            break

        if type_haystack == "essay" and num_haystack > len(haystack):
            num_haystack = len(haystack)
            break

        num_haystack += incremental

    # print("Num haystack:", num_haystack)

    # Generate samples
    for index in tqdm(
        range(num_samples),
        desc=f"Generating synthetic samples: {type_haystack} | {max_seq_length}",
    ):
        used_haystack = num_haystack
        while True:
            try:
                input_text, answer, query = generate_input_output(
                    used_haystack,
                    haystack,
                    type_haystack=type_haystack,
                    num_needle_k=num_needle_k,
                    type_needle_k=type_needle_k,
                    num_needle_v=num_needle_v,
                    type_needle_v=type_needle_v,
                    template=template,
                    num_needle_q=num_needle_q,
                    random_seed=random_seed,
                )
                length = len(TOKENIZER(input_text).input_ids) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
                # ruff: noqa
            except:
                if used_haystack > incremental:
                    used_haystack -= incremental

        if remove_newline_tab:
            input_text = " ".join(
                input_text.replace("\n", " ").replace("\t", " ").strip().split()
            )

        formatted_output = {
            "index": index,
            "input": input_text,
            "outputs": answer,
            "length": length,
            "max_length": max_seq_length,
            "gen_prefix": f"The special magic {type_needle_v[:-1]} for {query} mentioned in the provided text is"
            if num_needle_q * num_needle_v == 1
            else f"The special magic {type_needle_v} for {query} mentioned in the provided text are",
        }
        if formatted_output["outputs"][0] not in formatted_output["input"]:
            assert False, (
                f"Needle not in input: {formatted_output}. Something went wrong."
            )
        write_jsons.append(formatted_output)
    return write_jsons


@cache
def get_haystack(
    type_haystack: Literal["essay", "repeat", "needle"],
) -> Union[list[str], str]:
    NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
    if type_haystack == "essay":
        essay = datasets.load_dataset("baber/paul_graham_essays", split="train")["text"]
        essay = " ".join(essay)
        haystack = re.sub(r"\s+", " ", essay).split(" ")
    elif type_haystack == "repeat":
        haystack = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    elif type_haystack == "needle":
        haystack = NEEDLE
    else:
        raise NotImplementedError(f"{type_haystack} is not implemented.")
    return haystack