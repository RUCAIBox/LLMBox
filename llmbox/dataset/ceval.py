from logging import getLogger

from .generation_dataset import GenerationDataset
from .enum import CEVAL_TRANS, CEVAL_SUBJECTS
from ..metric import Em

logger = getLogger(__name__)

import re

class Ceval(GenerationDataset):
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

    instruction = "以下是中国关于{}考试的单项选择题，请选出其中的正确答案。"
    example_set = "dev"
    evaluation_set = "val"
    load_args = ("ceval/ceval-exam",)
    metrics = [Em()]
    categorized_subsets = CEVAL_SUBJECTS

    def __init__(self, args, model, subset_name: str):
        self.instruction = self.instruction.format(CEVAL_TRANS[subset_name])
        self.task = subset_name
        self.extra_model_args = dict(stop=["\n"]) if args.cot is None else dict()
        super().__init__(args, model, subset_name)

    def format_instance(self, instance):
        if instance["explanation"] is None or self.args.cot is None:
            target = instance["answer"]
        else:
            target = instance["explanation"][3:] + "\n" + "所以答案是" + instance["answer"] + "。"
        source = instance["question"].strip() + "\n"
        for idx in range(4):
            option = chr(ord('A') + idx)
            source += option + ". " + instance[option].strip() + "\n"
        source += "答案："
        if self.args.cot is not None:
            source += "让我们一步一步思考，\n1."
        return dict(
            source=source,
            target=" " + target
        )

    def post_processing(self, predictions):
        new_predictions = []
        for pred in predictions:
            extracted_answer = re.search(r"所以答案.([A-Z])", pred)
            if extracted_answer:
                new_pred = extracted_answer.group(1).strip()
            else:
                new_pred = pred.strip()
            new_predictions.append(new_pred)
        return new_predictions

    @property
    def references(self):
        return [instance["answer"] for instance in self.evaluation_data]
