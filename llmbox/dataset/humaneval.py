from ..metric import PassAtK
from .generation_dataset import GenerationDataset


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

    instruction = ""
    example_set = "test"
    evaluation_set = "test"
    load_args = ("openai_humaneval",)
    extra_model_args = dict(max_tokens=64, temperature=0.1)
    metrics = ""

    def __init__(self, args, model, subset_name=None):
        super().__init__(args, model, subset_name=subset_name)
        self.metrics = [PassAtK(k=args.pass_at_k)]

    def format_instance(self, instance):
        source_text = "Complete the following python function. Please only output the code for the completed function.\n\n\n" + instance[
            "prompt"]
        target_text = instance["canonical_solution"]
        return dict(source=source_text, target=target_text)

    @staticmethod
    def post_processing(predictions):
        return [prediction.lstrip("\n").split("\n\n")[0] for prediction in predictions]

    @property
    def references(self):
        return [
            instance["prompt"] + "{pred}" + "\n" + instance["test"] + "\n" + f"check({instance['entry_point']})"
            for instance in self.evaluation_data
        ]
