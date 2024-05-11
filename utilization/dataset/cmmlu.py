from functools import cached_property
from logging import getLogger

from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)


class Cmmlu(MultipleChoiceDataset):
    """The dataset of CMMLU.

    CMMLU: Measuring massive multitask language understanding in Chinese by Haonan Li and Yixuan Zhang and Fajri Koto and Yifei Yang and Hai Zhao and Yeyun Gong and Nan Duan and Timothy Baldwin.

    Example:
        "Question": "在农业生产中被当作极其重要的劳动对象发挥作用，最主要的不可替代的基本生产资料是",
        "A": "农业生产工具",
        "B": "土地",
        "C": "劳动力",
        "D": "资金",
        "Answer": "B"
    """

    instruction = "以下是关于({{subset_name}})的请单项选择题，直接给出正确答案的选项。\n\n题目：{{Question|trim}}{{'\n' + options if options}}\n答案："
    evaluation_set = "test"
    example_set = "dev"
    load_args = ("haonan-li/cmmlu",)

    def format_instance(self, instance):
        instance["target_idx"] = ord(instance["Answer"]) - ord('A')
        instance["options"] = [instance[op] for op in ("A", "B", "C", "D")]
        return instance

    def calculate_metric(self, predictions):
        results, score_lists = super().calculate_metric(predictions)
        return results, score_lists

    @cached_property
    def references(self):
        return [ord(instance["Answer"]) - ord('A') for instance in self.evaluation_data]
