from .metric import Metric
import openai
import time
from logging import getLogger
import re
import numpy as np
from tqdm import tqdm

logger = getLogger(__name__)

JUDGE_PROMPT = {"system_prompt": "You are a helpful assistant.", "prompt_template": "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]", "output_format": "[[rating]]"}

score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

class Mt_bench(Metric):
    r""" using strong LLMs as judges to evaluate models.

        Return:
            "Mt_bench score": float
        """

    def __call__(self, predictions, references):
        ratings = []
        for pred, refer in tqdm(zip(predictions, references), desc="Judging", total=len(predictions)):
            user_prompt = JUDGE_PROMPT["prompt_template"].format(question=refer, answer=pred)
            system_prompt = JUDGE_PROMPT["system_prompt"]
            for _ in range(5):
                try:
                    message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=message, temperature=0, max_tokens=2048)["choices"][0]["message"]["content"]
                    match = re.search(score_pattern, response)
                    if not match:
                        match = re.search(score_pattern_backup, response)

                    if match:
                        ratings.append(eval(match.groups()[0]))
                    else:
                        logger.warning(f"Fail to extract score from the response: {response}")
                        ratings.append(-1)
                    break

                except openai.error.RateLimitError:
                    logger.warning("Receive openai.error.RateLimitError, retrying...")
                    time.sleep(10)
                except openai.error.AuthenticationError as e:
                    raise e
                except openai.error.InvalidRequestError as e:
                    raise e
                except Exception as e:
                    logger.warning(f"Receive {e.__class__.__name__}: {str(e)}")
                    logger.warning("retrying...")
                    time.sleep(1)

        score_list = np.array(ratings)
        self._last_score_lists = {'Mt_bench score': score_list}
        filtered_arr = score_list[score_list != -1]
        return {'Mt_bench score': np.mean(filtered_arr) if len(filtered_arr) > 0 else -1}

