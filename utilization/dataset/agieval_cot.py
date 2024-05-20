import re
from functools import cached_property
from logging import getLogger

from ..metric import Em
from ..utils import math_equiv
from .dataset_enum import (
    AGIEVAL_EN_CLOZE_TASKS, AGIEVAL_EN_PROMPT_TASKS, AGIEVAL_EN_QA_TASKS, AGIEVAL_MULTI_ANSWERS_TASKS,
    AGIEVAL_NO_LETTER_CHOICE_TASKS, AGIEVAL_ZH_CLOZE_TASKS, AGIEVAL_ZH_PROMPT_TASKS, AGIEVAL_ZH_QA_TASKS
)
from .generation_dataset import GenerationDataset

logger = getLogger(__name__)

# `options_text` does not follow the standard MultipleChoiceDataset format,
# because there might be multiple correct answers in the AGIEval dataset.
INSTRUCTIONS = {
    "mcq_zh_nocot_zero_shot": "{passage}问题：{question} 选项：{options_text}\n答案：从A到{max_option_letter}，我们应选择",
    "mcq_zh_nocot_few_shot": "问题. {passage} {question}\n从以下选项中选择：{options_text}\n答案是",
    "mcq_zh_cot_zero_shot": "{passage}问题：{question} 选项：{options_text}\n答案：从A到{max_option_letter}，我们应选择什么？让我们逐步思考：",
    "mcq_zh_cot_few_shot": "问题. {passage} {question}\n从以下选项中选择：{options_text}\n问题的解析：",
    "mcq_en_nocot_zero_shot":
    "{passage}Q: {question} Answer Choices: {options_text}\nA: Among A through {max_option_letter}, the answer is",
    "mcq_en_nocot_few_shot":
    "Question. {passage} {question}\Choose from the following options: {options_text}\nThe answer is therefore",
    "mcq_en_cot_zero_shot": "{passage}Q: {question} Answer Choices: {options_text}\nLet's think step by step.",
    "mcq_en_cot_few_shot":
    "Question. {passage} {question}\nChoose from the following options: {options_text}\nExplanation for Problem:",
    "gen_zh_nocot_zero_shot": "{passage}问题：{question}\n答案：",
    "gen_zh_nocot_few_shot": "问题. {passage} {question}\n答案是",
    "gen_zh_cot_zero_shot": "{passage}问题：{question}\n答案：让我们逐步思考",
    "gen_zh_cot_few_shot": "问题. {passage} {question}\n问题的解析：",
    "gen_en_nocot_zero_shot": "{passage}Q: {question}\nA: The answer is",
    "gen_en_nocot_few_shot": "Question. {passage} {question}\nThe answer is therefore",
    "gen_en_cot_zero_shot": "{passage}Q: {question}\nA: Let's think step by step",
    "gen_en_cot_few_shot": "Question. {passage} {question}\nExplanation for Problem:",
}

TARGETS = {
    "zh_cot_few_shot": " {explanation}\n答案是 {label}",
    "en_cot_few_shot": " {explanation}\nThe answer is therefore {label}",
}


class Agieval_cot(GenerationDataset):
    """The dataset of AGIEval, as a generation dataset.
    We support zero shot no CoT, few shot no CoT, few shot with CoT in this dataset.

    AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models by Wanjun Zhong and Ruixiang Cui and Yiduo Guo and Yaobo Liang and Shuai Lu and Yanlin Wang and Amin Saied and Weizhu Chen and Nan Duan.

    Example:
        "passage": "__张祖传__。[明]张岳。张祖，字彦宗，以字行。十三岁，父祖继殁，独奉母以居。洪武改元，闽中法令严核，绳吏之法尤峻。惮应役者邀祖斩右大指以自黜。祖疑之，入白母。母曰：“法可避也，指斩不可复续，盍去诸？”遂避匿。未几，斩指事觉，诏逮捕戍边。犯者言张某始与某辈约如此。逮久弗获。会天变肆赦，乃归。室中空虚，至系马槛牛，毁斗桶为薪。念非力学无以树门户，于是决意习儒业。是时，诏民田八顷以上家，择子若①孙一人为吏。县檄至，祖挥之弗受，执卷奋曰：“吾而吏耶？”令白按察司，复檄祖往，固弗受如县。使者熟视之，曰：“君，我辈中人也，勿辱于县。”遂挟以去。祖既通儒术，兼晓九章算法。时方行方田②于令，即以其事属之。文案盈几，祖精勤不舍，昼夜栉理而错画之，皆有绪可按据。建文时，祖为吏部吏。未几，云南布政张公紞召入为尚书，于属吏多所更易，独言张某老成，守法不易也。时帝方与方孝孺辈讲求古治，经济之事多变太祖旧章，章奏日下吏部。祖密言于紞曰：“高皇帝起布衣，有天下，立法创制，规模远矣。为治当责实效。今法制已定，日有变更，未必胜于旧，徒使异议者以为口实，盍先其急者？”紞深然之，而夺于群议，不能用。会添设京卫知事一员，诏吏部选可者。紞曰：“无逾祖矣。”授留守知事。及靖难师渡江，祖为安吉县丞。紞被谴自经③，舁尸归，属吏无敢往视，祖独往经理其殡。殡毕，哭奠而去。时人义之。安吉在万山中，向多逋民④，隐田不以自实，财赋甚少。祖至，清勤自持，敬礼贤士大夫，与讲究磨砺。在职九年，稽核财赋，修筑陂塘圩岸，不可胜计。逋民隐田者令以占籍⑤输税，免其罪。声称著闻，以最荐升湖广按察司经历。行至吴桥卒，惟一子扶丧归。（摘编自《小山类稿》）。",
        "question": "下列对原文内容的概括与分析，不正确的一项是                                 (   )",
        "options": ["(A)张祖为逃避服役而断指出走，遇赦后见家境衰败，于是决定发愤读书以振兴家门。", "(B)建文年间张祖在吏部做小吏，上司张紞非常赏识他，认为他办事老成，笃守法令。", "(C)吏部尚书张紞自杀后，属吏中只有张祖敢出面料理丧事，当时的人认为他有情有义。", "(D)张祖任安吉县丞九年，因政绩卓著，考核获得最高等级，被推荐升任湖广按察司经历。"],
        "label": "A",
        "other": {"source": "2014年福建省高考语文试题()"},
        "explanation": "根据原文，张祖并非为逃避服役而断指出走，而是因为有人邀请他砍掉右手大指以便避免服役。但张祖并未轻信他人，他先向母亲请示，母亲建议他逃避，于是张祖隐瞒了自己的行踪。因此，选项(A)所述并不准确。选项(B)正确概括了建文年间张祖在吏部做小吏的情况，并指出了上司张公紞非常赏识他。选项(C)也正确描述了张紞自杀后，张祖成为属吏中唯一料理丧事的人，受到当时人们的赞誉和尊敬。选项(D)则准确地概括了张祖在任安吉县丞期间的表现和考核结果。"
    """

    instruction = ""
    example_set = "dev"
    evaluation_set = "test"
    load_args = ("RUCAIBox/agieval",)
    metrics = [Em()]
    supported_cot = ["base"]

    def init_arguments(self):
        self.extra_model_args = dict(stop=["\n"]) if self.cot is None else dict()
        text = ""
        text += "gen" if self.subset_name in AGIEVAL_NO_LETTER_CHOICE_TASKS else "mcq"
        text += "_zh" if self.subset_name in AGIEVAL_ZH_PROMPT_TASKS else "_en"
        text += "_cot" if self.cot else "_nocot"
        text += "_few_shot" if self.max_num_shots > 0 else "_zero_shot"
        self.instruction = INSTRUCTIONS[text]
        self.target_template = TARGETS.get(text[4:], " {label}")

    def format_instance(self, instance):
        if instance.get("options") is not None:
            instance["options_text"] = self._choice_to_str(instance["options"])
            instance["max_option_letter"] = self._max_choice_letter(instance["options"])
        return instance

    def post_processing(self, predictions):
        new_predictions = []
        for pred in predictions:
            if pred is None:
                new_predictions.append("")
                continue
            new_pred = self.post_process(pred)
            if new_pred is not None:
                while new_pred.endswith("."):
                    new_pred = new_pred[:len(new_pred) - 1]
                if self.subset_name in AGIEVAL_NO_LETTER_CHOICE_TASKS:
                    new_pred = math_equiv._strip_string(new_pred)
            else:
                new_pred = ""
            new_predictions.append(new_pred)

        return new_predictions

    @staticmethod
    def _find_choice(prediction):
        letter_set = {"A", "B", "C", "D", "E", "F"}
        for c in prediction:
            if c in letter_set:
                return c
        return ""

    @staticmethod
    def _choice_to_str(choices):
        target_str = ""
        for option in choices:
            target_str += option.strip() + " "
        return target_str.strip()

    @staticmethod
    def _max_choice_letter(choices):
        return chr(ord('A') + len(choices) - 1)

    @cached_property
    def references(self):
        if self.subset_name not in AGIEVAL_NO_LETTER_CHOICE_TASKS:
            return [[str(instance["label"])] for instance in self.evaluation_data]  # type: ignore
        else:
            return [
                [math_equiv._strip_string(str(instance["label"]))]  # type: ignore
                for instance in self.evaluation_data
            ]

    # The following code comes from https://github.com/ruixiangcui/AGIEval/tree/main/src
    # Copyright (c) Microsoft Corporation.
    # Licensed under the MIT license.

    def extract_last_line(self, string):
        lines = string.split('\n')
        for item in lines[::-1]:
            if item.strip() != "":
                string = item
                break
        return string

    def remove_few_shot_prefix(self, string: str):
        prefix_list = ["The answer is therefore", "答案是"]
        for prefix in prefix_list:
            if string.startswith(prefix):
                string = string[len(prefix):].strip()
            elif prefix in string:
                index = string.rfind(prefix)
                if index >= 0:
                    string = string[index + len(prefix):].strip()
        return string

    def try_parse_few_shot_qa_single_answer(self, string, language='en'):
        if self.max_num_shots > 0 and self.cot is not None:
            string = self.extract_last_line(string)
        if language == 'en':
            pattern = "answer is .*?([A-G])"
            match = re.search(pattern, string)
        elif language == 'zh':
            pattern = "答案是.*?([A-G])"
            match = re.search(pattern, string)
        if match:
            return match.group(1)
        else:
            return None

    def try_parse_few_shot_pattern(self, string: str):
        if self.max_num_shots > 0 and self.cot is not None:
            string = self.extract_last_line(string)
        if self.subset_name in AGIEVAL_ZH_CLOZE_TASKS:
            return string.startswith("答案是")
        elif self.subset_name in AGIEVAL_EN_CLOZE_TASKS:
            return string.startswith("The answer is therefore")
        elif self.subset_name in AGIEVAL_ZH_QA_TASKS:
            pattern = "答案是.*?([A-G])"
            match = re.search(pattern, string)
            return match is not None
        elif self.subset_name in AGIEVAL_EN_QA_TASKS:
            pattern = "answer is .*?([A-G])"
            match = re.search(pattern, string)
            return match is not None
        return False

    def parse_few_shot_qa_single_answer(self, string, language='en'):
        answer = self.try_parse_few_shot_qa_single_answer(string, language)
        if answer is None:
            return self._find_choice(string)
        else:
            return answer

    def parse_math_answer(self, raw_string):
        if self.max_num_shots > 0 and self.cot is not None:
            raw_string = self.extract_last_line(raw_string)
        if self.max_num_shots > 0:
            raw_string = self.remove_few_shot_prefix(raw_string)
            return raw_string

        def remove_boxed(s):
            left = "\\boxed{"
            try:
                assert s[:len(left)] == left
                assert s[-1] == "}"
                answer = s[len(left):-1]
                if "=" in answer:
                    answer = answer.split("=")[-1].lstrip(" ")
                return answer
            except:
                return None

        def last_boxed_only_string(string: str):
            idx = string.rfind("\\boxed")
            if idx < 0:
                idx = string.rfind("\\fbox")
                if idx < 0:
                    return None
            i = idx
            right_brace_idx = None
            num_left_braces_open = 0
            while i < len(string):
                if string[i] == "{":
                    num_left_braces_open += 1
                if string[i] == "}":
                    num_left_braces_open -= 1
                    if num_left_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1

            if right_brace_idx == None:
                retval = None
            else:
                retval = string[idx:right_brace_idx + 1]

            return retval

        def get_answer_with_dollar_sign(s):
            first_pattern = r"\$(.*)\$"
            last_match = None
            matches = re.findall(first_pattern, s)
            if matches:
                last_match = matches[-1]
                if "=" in last_match:
                    last_match = last_match.split("=")[-1].lstrip(" ")
            return last_match

        def get_answer_without_dollar_sign(s):
            last_match = None
            if "=" in s:
                last_match = s.split("=")[-1].lstrip(" ").rstrip(".")
                if "\\n" in last_match:
                    last_match = last_match.split("\\n")[0]
            else:
                pattern = r"(?:\\$)?\d+(?:\.\d+)?(?![\w\d])"
                matches = re.findall(pattern, s)
                if matches:
                    last_match = matches[-1]
            return last_match

        raw_string = self.remove_few_shot_prefix(raw_string)
        if "\\boxed" in raw_string:
            answer = remove_boxed(last_boxed_only_string(raw_string))
        else:
            answer = get_answer_with_dollar_sign(raw_string)
            if not answer:
                answer = get_answer_without_dollar_sign(raw_string)
        return answer

    def parse_qa_multiple_answer(self, string):
        if self.max_num_shots > 0 and self.cot is not None:
            string = self.extract_last_line(string)
        pattern = r"\(*([A-F])\)*"
        match = re.findall(pattern, string)
        if match:
            answers = sorted(set(match))
            if self.subset_name == "gaokao-physics" and len(answers) == 1:
                return answers[0]
            return answers.__str__()
        return ""

    def post_process(self, prediction):
        if self.subset_name in AGIEVAL_NO_LETTER_CHOICE_TASKS:
            return self.parse_math_answer(prediction)

        if self.subset_name in AGIEVAL_MULTI_ANSWERS_TASKS:
            return self.parse_qa_multiple_answer(prediction)

        if self.max_num_shots == 0:
            answer = self._find_choice(prediction)
            return answer

        language = "en" if self.subset_name in AGIEVAL_EN_PROMPT_TASKS else "zh"
        return self.parse_few_shot_qa_single_answer(prediction, language)
