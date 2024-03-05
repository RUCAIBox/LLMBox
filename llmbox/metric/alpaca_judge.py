import re
from logging import getLogger
from typing import Dict

import ast
import logging
import numpy as np
import openai
from tqdm import tqdm

from ..model import load_model
from ..utils import ModelArguments
from .metric import Metric

logger = getLogger(__name__)


class Alpaca_judge(Metric):

    def __call__(self, predictions, references):
        model_args = ModelArguments(
            model_name_or_path="gpt-3.5-turbo-instruct", max_tokens=100, temperature=0, openai_api_key=openai.api_key
        )
        model = load_model(model_args)
        model.set_generation_args()
        responses = []
        n_occurences = ["instruction", "output_1", "output_2"]
        for pred, refer in tqdm(zip(predictions, references), desc="Judging", total=len(predictions)):
            n_replaces = [refer["instruction"], pred, refer["output"]]
            current_prompt = template_prompt
            for to_format, to_replace in zip(n_occurences, n_replaces):
                current_prompt = current_prompt.replace("{" + to_format + "}", to_replace)
            responses.extend(model.generation([current_prompt]))

        preferences = np.array(Alpaca_judge._get_preferences(responses))
        is_preference = (preferences >= 1) & (preferences <= 2)
        n_not_pair = np.sum(~is_preference)
        if n_not_pair > 0:
            logging.info(f"drop {n_not_pair} outputs that are not preferences")

        preferences = preferences[is_preference] - 1
        score_list = preferences
        self._last_score_lists = {"win_rate": score_list, "standard_error": score_list}
        print(score_list)
        return {
            "win_rate": np.mean(score_list) * 100,
            "standard_error": np.std(score_list) * 100 / np.sqrt(len(score_list))
        }

    @staticmethod
    def _get_preferences(predictions):
        preferences = []
        for pair_rank in predictions:
            preferences.append(Alpaca_judge._ranking_parser(pair_rank))
        return preferences

    @staticmethod
    def _ranking_parser(pair_rank):
        """Preference will be 1.5 if output_1 == output_2, 1 if output_1 is preferred, and 2 if output_2"""
        try:
            pair_rank = ast.literal_eval(pair_rank)
            rank1 = pair_rank[0]["rank"]
            rank2 = pair_rank[1]["rank"]
            if rank1 == rank2:
                return 1.5
            return [c for c in pair_rank if c["model"] == "model_2"][0]["rank"]
        except Exception as e:
            logging.error(f"{e}\nRank1: {rank1}\nRank2: {rank2}\n"
                          "You must manually fix the score pair.")
            return np.nan


template_prompt = """<|im_start|>system
You are a helpful assistant, that ranks models by the quality of their answers.
<|im_end|>
<|im_start|>user
I want you to create a leaderboard of different of large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

Here is the prompt:
{
    "instruction": \"""{instruction}\""",
}

Here are the outputs of the models:
[
    {
        "model": "model_1",
        "answer": \"""{output_1}\"""
    },
    {
        "model": "model_2",
        "answer": \"""{output_2}\"""
    }
]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
    {'model': <model-name>, 'rank': <model-rank>},
    {'model': <model-name>, 'rank': <model-rank>}
]

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give.
<|im_end|>"""