from logging import getLogger
from typing import List, Tuple

from .multiple_choice_dataset import MultipleChoiceDataset
from .enum import CMMLU_NAME_TRANS

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

    instruction = "以下是关于({})的单项选择题，请直接给出正确答案的选项。"
    evaluation_set = "test"
    example_set = "dev"
    load_args = ("haonan-li/cmmlu",)

    def __init__(self, args, model, subset_name: str):
        self.instruction = self.instruction.format(CMMLU_NAME_TRANS[subset_name])
        if args.ranking_type.startswith("ppl"):  # ppl or ppl_no_option
            self.source_prefix = "题目："
        elif args.ranking_type == "prob":
            self.source_prefix = ""
        super().__init__(args, model, subset_name)

    def format_instance(self, instance):
        options = list(map(lambda op: " " + op, [instance[chr(ord('A') + _)] for _ in range(4)]))
        return dict(
            source=self.source_prefix + instance["Question"].strip(),
            source_postfix="\n答案是",
            target_idx=ord(instance["Answer"]) - ord('A'),
            options=options,
        )

    def calculate_metric(self, predictions):
        results, score_lists = super().calculate_metric(predictions)
        return results, score_lists

    @property
    def references(self):
        return [ord(instance["Answer"]) - ord('A') for instance in self.evaluation_data]
