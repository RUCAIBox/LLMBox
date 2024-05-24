import json
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)

FIVE_SHOT_INSTRUCTION = ""

INSTRUCTIONS = {
    "five_shot":
    """{{ 'Here are some example questions from experts. Answer the final question yourself, following the format of the previous questions exactly.\n' if example_idx == 0 }}Question: {{ question }}{{ '\nChoices:\n' + options if options }}\n{{ 'The correct answer is' if example_idx >= 0 else ('When you' + "'" + 're ready to answer, please use the format "The correct answer is (insert answer here)".') }}""",
    "zero_shot":
    """What is the correct answer to this question: {{ question }}{{ '\n\nChoices:\n' + options if options }}\n\nFormat your response as follows: "The correct answer is (insert answer here)".""",
    "five_shot_cot":
    """{{ 'Here are some example questions from experts. An explanation is given before the final answer. Answer the final question yourself, giving your reasoning beforehand.\n' if example_idx == 0 }}Question: {{ question }}{{ '\nChoices:\n' + options if options }}\n{{ 'The correct answer is' if example_idx >= 0 else ('Give step by step reasoning before you answer, and when you' + "'" + 're ready to answer, please use the format "The correct answer is (insert answer here)":\n') }}""",
    "zero_shot_cot":
    """What is the correct answer to this question: {{ question }}{{ '\n\nChoices:\n' + options if options }}\nLet's think step by step: __SEPARATOR__\n\nBased on the above, what is the single, most likely answer choice? Answer in the format "The correct answer is (insert answer here)".""",
    "retrieval": "{{ bing }}\n\nQuestion: {{ question }}{{ '\n\nChoices:\n' + options if options }}"
}


class GPQA(MultipleChoiceDataset):
    """The dataset of GPQA.
    """

    instruction = None
    evaluation_set = "train"
    example_set = None
    load_args = ("Idavidrein/gpqa", "gpqa_main")
    supported_cot = ["base", "retrieval", "retrieval_content"]
    extra_model_args = {"stop": ["<|eot_id|>", "<|start_header_id|>"]}

    def init_arguments(self):

        if self.sample_num > 1 and (self.max_num_shots == 0 or self.cot is None):
            logger.warning(
                "Self-consistency is only available for 5-shot CoT. Automatically set max_num_shots to 5 and cot to base."
            )
            self.max_num_shots = 5
            self.cot = "base"

        if self.max_num_shots > 0 and self.cot in ["retrieval", "retrieval_content"]:
            logger.warning("Open-book baselines do not support few-shot. Automatically set max_num_shots to 0.")
            self.max_num_shots = 0

        if self.max_num_shots == 0 and self.cot == "base":
            self.extra_model_args["multi_turn"] = True

        if self.max_num_shots != 0 and self.max_num_shots != 5:
            logger.warning("Only 0-shot and 5-shot are supported for GPQA, but got %d-shot.", self.max_num_shots)

        if self.model_evaluation_method == "generation":
            if self.sample_num > 1:
                self.extra_model_args["temperature"] = 0.7
                self.extra_model_args["max_tokens"] = 1000
            else:
                self.extra_model_args["temperature"] = 0.0
                self.extra_model_args["max_tokens"] = 1000

        if self.max_num_shots > 0 and self.cot == "base":
            self.instruction = INSTRUCTIONS["five_shot_cot"]
        elif self.max_num_shots > 0:
            self.instruction = INSTRUCTIONS["five_shot"]
        elif self.max_num_shots == 0 and self.cot == "base":
            self.instruction = INSTRUCTIONS["zero_shot_cot"]
        elif self.max_num_shots == 0 and self.cot in ["retrieval", "retrieval_content"]:
            self.instruction = INSTRUCTIONS["retrieval"]
        elif self.max_num_shots == 0:
            self.instruction = INSTRUCTIONS["zero_shot"]

    @staticmethod
    def read_file(base_path, search_path):
        examples_json = None
        for i in range(len(search_path)):
            test_path = base_path / "/".join(search_path[i:])
            if test_path.is_file() and test_path.suffix == ".json":
                examples_json = base_path / "/".join(search_path[i:])
                break

        if examples_json is None:
            return None
        else:
            return open(examples_json, "r").read()

    def load_raw_dataset(
        self, dataset_path: Optional[str], subset_name: Optional[str], evaluation_set: str, example_set: Optional[str]
    ):

        super().load_raw_dataset(dataset_path, subset_name, evaluation_set, None)

        self.bing_data = None

        if example_set is not None:
            example_set_path = Path(example_set)

            if self.max_num_shots > 0 and self.cot in ["base", None]:
                SEARCH_PATH = ["prompts", "chain_of_thought_examples.json"]
                data = self.read_file(example_set_path, SEARCH_PATH)
                if data is None:
                    raise FileNotFoundError(
                        "Example set not found. Please provide the correct path to the example set through `--example_set path/to/prompts`."
                    )
                self.example_data = json.loads(data)["questions"]

            elif self.max_num_shots == 0 and self.cot in ["retrieval", "retrieval_content"]:
                if self.cot == "retrieval":
                    SEARCH_PATH = ["prompts", "bing-search-cot.txt"]
                else:
                    SEARCH_PATH = ["prompts", "bing-search-cot-content.txt"]
                data = self.read_file(example_set_path, SEARCH_PATH)
                if data is None:
                    raise FileNotFoundError(
                        "Chain-of-thought propmts for open-book not found. Please provide the correct path to the example set through `--example_set path/to/prompts`."
                    )
                self.bing_data = data

    def format_instance(self, instance):
        instance["bing"] = self.bing_data
        if "choices" in instance:
            # https://github.com/idavidrein/gpqa/blob/main/prompts/chain_of_thought_examples.json
            instance["options"] = list(instance["choices"].values())
            instance["target_idx"] = ord(instance["correct_answer"]) - ord("A")
        else:
            # https://huggingface.co/datasets/Idavidrein/gpqa
            instance["question"] = instance["Question"]
            instance["options"] = [
                instance["Incorrect Answer 1"], instance["Incorrect Answer 2"], instance["Incorrect Answer 3"],
                instance["Correct Answer"]
            ]
            order = self.shuffle_options(instance["options"])
            instance["target_idx"] = order.index(3)
        return instance

    def post_processing(self, predictions: Union[List[Tuple[float, int]], List[List[int]], List[str]]) -> List[int]:
        if isinstance(predictions[0], (tuple, list)):
            predictions = [pred[1] for pred in predictions]
        return MultipleChoiceDataset.post_processing(self, predictions)

    @cached_property
    def references(self):
        return [instance["target_idx"] for instance in self.evaluation_data]
