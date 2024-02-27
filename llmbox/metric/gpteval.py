from .metric import Metric
from logging import getLogger
import re
import numpy as np
from tqdm import tqdm
from ..utils import ModelArguments
from ..model import load_model

logger = getLogger(__name__)

JUDGE_PROMPT = "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
JUDGE_PROMPT_MATH = "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer_1}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"
JUDGE_PROMPT_MT = "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You evaluation should focus on the assistant's answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n<|The End of Assistant A's Conversation with User|>"
JUDGE_PROMPT_MATH_MT = "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. You evaluation should focus on the assistant's answer to the second question. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n<|The Start of Reference Answer|>\n\n### User:\n{question_1}\n\n### Reference answer:\n{ref_answer_1}\n\n### User:\n{question_2}\n\n### Reference answer:\n{ref_answer_2}\n\n<|The End of Reference Answer|>\n\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n<|The End of Assistant A's Conversation with User|>"

score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
score_pattern_backup2 = re.compile("(\d+\.?\d*)")


class GPTEval(Metric):
    r"""Using strong LLMs as judges to evaluate models. (Single/Multi Turn)

        Return:
            "GPT-Eval": float
        """

    def __init__(self, multi_turn=False):
        self.multi_turn = multi_turn

    def __call__(self, predictions, references):
        model_args = ModelArguments(
            model_name_or_path="gpt-3.5-turbo", # use it to judge the model.
            max_tokens=1024,
            temperature=0,
        )
        model = load_model(model_args)
        model.set_generation_args()
        responses = []
        for pred, refer in tqdm(zip(predictions, references), desc="Judging", total=len(predictions)):
            if "ref_answer_1" not in refer:
                user_prompt = JUDGE_PROMPT_MT.format(
                    question_1=refer["question_1"],
                    answer_1=pred[0],
                    question_2=refer["question_2"],
                    answer_2=pred[1]
                ) if self.multi_turn else JUDGE_PROMPT.format(
                    question=refer["turns"][0], answer=pred
                )
            else:
                user_prompt = JUDGE_PROMPT_MATH_MT.format(
                    question_1=refer["question_1"],
                    answer_1=pred[0],
                    ref_answer_1=refer["ref_answer_1"],
                    question_2=refer["question_2"],
                    answer_2=pred[1],
                    ref_answer_2=refer["ref_answer_2"]
                ) if self.multi_turn else JUDGE_PROMPT_MATH.format(
                    question=refer["turns"][0], ref_answer_1=refer["ref_answer_1"], answer=pred
                )
            responses.extend(model.generation([user_prompt]))

        ratings = []
        for response in responses:
            match = re.search(score_pattern, response)
            if not match:
                match = re.search(score_pattern_backup, response)
                if not match:
                    match = re.findall(score_pattern_backup2, response)
                    if match:
                        rating = eval(match[-1])
                else:
                    rating = eval(match.groups()[0])
            else:
                rating = eval(match.groups()[0])

            if match:
                ratings.append(min(max(rating, 1), 10))
            else:
                ratings.append(1)
                logger.warning(f"Failed to extract rating from response: {response}")

        score_list = np.array(ratings)
        self._last_score_lists = {'GPT-Eval': score_list}
        return {'GPT-Eval': np.mean(score_list)}
