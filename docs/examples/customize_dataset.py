import os
import sys

sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utilization import DatasetArguments, ModelArguments, get_evaluator, register_dataset
from utilization.dataset import GenerationDataset


@register_dataset(name="my_data")
class MyData(GenerationDataset):

    instruction = "Reply to my message: {input}\nReply:"
    metrics = []

    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        self.evaluation_data = [
            {
                "input": "Hello",
                "target": "Hi"
            },
            {
                "input": "How are you?",
                "target": "I'm fine, thank you!"
            },
        ]
        self.example_data = [{"input": "What's the weather like today?", "target": "It's sunny today."}]

    def format_instance(self, instance: dict) -> dict:
        return instance

    @property
    def references(self):
        return [i["target"] for i in self.evaluation_data]


evaluator = get_evaluator(
    model_args=ModelArguments(model_name_or_path="gpt-4o"),
    dataset_args=DatasetArguments(
        dataset_names=["my_data"],
        num_shots=1,
        max_example_tokens=2560,
    ),
)
evaluator.evaluate()
