from ..metric import GPTEvalMT
from .generation_dataset import GenerationDataset


class Mt_bench(GenerationDataset):
    """The dataset of Mt_bench.

    MT-bench is a set of challenging multi-turn open-ended questions for evaluating chat assistants. Use MT-bench questions and prompts to evaluate the models with LLM-as-a-judge.

    Example:
        prompt_id: 15896981
        prompt: ["Imagine you are participating in a race with a group of people. If you have just overtaken the second person, what's your current position? Where is the person you just overtook?", "If the \"second person\" is changed to \"last person\" in the above question, what would the answer be?"]
        category: "reasoning"
        reference: ["You are in second place.", "Uncertain."]
    """

    instruction = ""
    example_set = ""
    evaluation_set = "train"
    load_args = ("HuggingFaceH4/mt_bench_prompts",)
    metrics = [GPTEvalMT()]

    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        super().load_raw_dataset(dataset_path, subset_name, evaluation_set, example_set)
        new_evaluation_data = []
        for instance in self.evaluation_data:
            if len(instance["reference"]) != 0:
                new_evaluation_data.append({
                    "question_1": instance["prompt"][0],
                    "question_2": instance["prompt"][1],
                    "ref_answer_1": instance["reference"][0],
                    "ref_answer_2": instance["reference"][1]
                })
            else:
                new_evaluation_data.append({
                    "question_1": instance["prompt"][0],
                    "question_2": instance["prompt"][1]
                })
        self.evaluation_data = new_evaluation_data


    def format_instance(self, instance):
        return dict(
            source=instance["question_1"] + "\n" + instance["question_2"],
            target="",
        )

    @property
    def references(self):
        return self.evaluation_data
