from functools import cached_property
from logging import getLogger

from .enum import CEVAL_SUBJECTS, CEVAL_TRANS
from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)


class Ceval(MultipleChoiceDataset):
    """The dataset of C-Eval.

    C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models by Huang, Yuzhen and Bai, Yuzhuo and Zhu, Zhihao and Zhang, Junlei and Zhang, Jinghan and Su, Tangjun and Liu, Junteng and Lv, Chuancheng and Zhang, Yikai and Lei, Jiayi and Fu, Yao and Sun, Maosong and He, Junxian.

    Example:
        "id": 1
        "question": "E47是人工合成的，由47个核苷酸组成的单链DNA分子，它可以催化两个DNA片段之间的连接。下列有关E47的分析，错误的是____"
        "A": "A与T的比例不一定相等"
        "B": "具有双螺旋结构"
        "C": "具有酶的功能"
        "D": "碱基序列决定其特异性"
        "answer": "B"
        "explanation": "1. E47是一个单链DNA分子，而双螺旋结构是由两条互补的DNA链通过碱基配对形成的，所以E47不具有双螺旋结构，B选项错误。"
    """

    instruction = "以下是中国关于{{subset_zh}}考试的单项选择题，请选出其中的正确答案。\n\n{{question|trim}}{{'\n'+options if options}}\n答案："
    example_set = "dev"
    evaluation_set = "val"
    load_args = ("ceval/ceval-exam",)
    categorized_subsets = CEVAL_SUBJECTS

    def init_arguments(self):
        self.subset_zh = CEVAL_TRANS[self.subset_name]

    def format_instance(self, instance):
        instance["subset_zh"] = self.subset_zh
        instance["options"] = ["A", "B", "C", "D"]
        instance["target_idx"] = ord(instance["answer"]) - ord('A')
        return instance

    def calculate_metric(self, predictions):
        results, score_lists = super().calculate_metric(predictions)
        return results, score_lists

    @cached_property
    def references(self):
        return [ord(instance["answer"]) - ord('A') for instance in self.evaluation_data]
