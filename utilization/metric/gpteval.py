import re
import time
from logging import getLogger

import numpy as np
import openai
from openai.error import InvalidRequestError
from tqdm import tqdm

from ..model import load_model
from ..utils import ModelArguments
from .metric import Metric

logger = getLogger(__name__)

SINGLE_JUDGE_PROMPT = "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
SINGLE_JUDGE_PROMPT_MATH = "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer_1}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
SINGLE_JUDGE_PROMPT_MT = "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You evaluation should focus on the assistant's answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n<|The End of Assistant A's Conversation with User|>"
SINGLE_JUDGE_PROMPT_MATH_MT = "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. You evaluation should focus on the assistant's answer to the second question. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n<|The Start of Reference Answer|>\n\n### User:\n{question_1}\n\n### Reference answer:\n{ref_answer_1}\n\n### User:\n{question_2}\n\n### Reference answer:\n{ref_answer_2}\n\n<|The End of Reference Answer|>\n\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n<|The End of Assistant A's Conversation with User|>"

PAIRWISE_JUDGE_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.\n\n[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"

score_patterns = [
    re.compile(r"\[\[(\d+\.?\d*)\]\]"),
    re.compile(r"\[(\d+\.?\d*)\]"),
    re.compile(r"(\d+\.?\d*)"),
]


class GPTEval(Metric):
    r"""Using strong LLMs as judges to evaluate models. (Single/Multi Turn)

        Return:
            "GPT-Eval": float
        """

    def __init__(self, multi_turn=False, type="single"):
        self.multi_turn = multi_turn
        self.type = type
        self.model_args = ModelArguments(
            model_name_or_path="gpt-3.5-turbo",  # use it to judge the model.
            max_tokens=1024,
            temperature=0,
            openai_api_key=openai.api_key
        )
        self.min_scoring = 1
        self.max_scoring = 10

    def __call__(self, predictions, references):

        # load gpteval model after the predictions of dataset are generated
        self.model = load_model(self.model_args)
        self.model.set_generation_args()

        if self.type == "single":
            ratings = self._get_single_ratings(predictions, references)
        elif self.type == "pairwise":
            ratings = self._get_pairwise_ratings(predictions, references)
        else:
            raise ValueError(f"{self.type} does not exists, please check the type")
        score_list = np.array(ratings)
        self._last_score_lists = {'GPT-Eval': score_list}
        return {'GPT-Eval': np.mean(score_list)}

    def _generation(self, prompt):
        try:
            return self.model.generation([prompt])
        except InvalidRequestError as e:  # continue to generate the response
            logger.warning(f"Failed to generate GPTEval response: {e}")
            return [str(self.min_scoring)]

    def _get_single_ratings(self, predictions, references):
        responses = []
        for pred, refer in tqdm(zip(predictions, references), desc="Judging", total=len(predictions)):
            if "ref_answer_1" not in refer:
                user_prompt = SINGLE_JUDGE_PROMPT_MT.format(
                    question_1=refer["question_1"], answer_1=pred[0], question_2=refer["question_2"], answer_2=pred[1]
                ) if self.multi_turn else SINGLE_JUDGE_PROMPT.format(question=refer["turns"][0], answer=pred)
            else:
                user_prompt = SINGLE_JUDGE_PROMPT_MATH_MT.format(
                    question_1=refer["question_1"],
                    answer_1=pred[0],
                    ref_answer_1=refer["ref_answer_1"],
                    question_2=refer["question_2"],
                    answer_2=pred[1],
                    ref_answer_2=refer["ref_answer_2"]
                ) if self.multi_turn else SINGLE_JUDGE_PROMPT_MATH.format(
                    question=refer["turns"][0], ref_answer_1=refer["ref_answer_1"], answer=pred
                )
            responses.extend(self._generation(user_prompt))

        ratings = []
        for response in responses:
            rating = None
            for pattern in score_patterns:
                matches = pattern.findall(response)
                if len(matches):
                    rating = float(matches[-1])
                    break

            if rating is not None:
                ratings.append(min(max(rating, self.min_scoring), self.max_scoring))
            else:
                ratings.append(self.min_scoring)
                logger.warning(f"Failed to extract rating from response: {response}")
        return ratings

    def _get_pairwise_ratings(self, predictions, references):
        responses = []
        for pred, refer in tqdm(zip(predictions, references), desc="Judging", total=len(predictions)):
            current_prompt = PAIRWISE_JUDGE_PROMPT.format(
                question=refer["instruction"], answer_a=refer["output"], answer_b=pred
            )
            responses.extend(self._generation(current_prompt))
            time.sleep(5)

        ratings = []
        for response in responses:
            match = pattern = r"\[\[(A|B|C)\]\]"
            match = re.search(pattern, response)
            if match:
                if match.group(1) == 'A':
                    ratings.append(0.0)
                elif match.group(1) == 'B':
                    ratings.append(1.0)
                else:
                    ratings.append(0.5)
            else:
                logger.warning(f"Failed to extract rating from response: {response}")
                ratings.append(0.5)
        return ratings
