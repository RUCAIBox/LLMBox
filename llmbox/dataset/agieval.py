from logging import getLogger

from .generation_dataset import GenerationDataset
from .enum import AGIEVAL_WORDS, AGIEVAL_EN_QA, AGIEVAL_ZH_QA, AGIEVAL_EN_CLOZE, AGIEVAL_ZH_CLOZE, AGIEVAL_MULTI_CHOICE, AGIEVAL_CHINESE_TASK, AGIEVAL_NO_LETTER_CHOICE
from ..utils import math_equiv
from ..metric import Em

logger = getLogger(__name__)

import re

class Agieval(GenerationDataset):
    """The dataset of AGIEval.

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

    def __init__(self, args, model, subset_name: str):
        self.task = subset_name
        self.shots = args.num_shots
        self.extra_model_args = dict(stop=["\n"]) if args.cot is None else dict()
        super().__init__(args, model, subset_name)

    def format_instance(self, instance):
        WORDS = [_[0 if self.task in AGIEVAL_CHINESE_TASK else 1] for _ in AGIEVAL_WORDS]
        passage = "" if instance["passage"] is None else instance["passage"]
        target = instance["label"]
        if self.task not in AGIEVAL_NO_LETTER_CHOICE:
            if self.shots == 0:
                source = passage + WORDS[0] + instance["question"] + " " + WORDS[1] + self._choice_to_str(instance["options"]) + "\n"
                if self.args.cot is None:
                    source += WORDS[2].format(self._max_choice_letter(instance["options"]))
                else:
                    source += WORDS[3].format(self._max_choice_letter(instance["options"]))
            else:
                source = WORDS[6] + passage + " " + instance["question"] + "\n" + WORDS[7] + self._choice_to_str(instance["options"]) + "\n"
                if self.args.cot is None:
                    source += WORDS[5]
                else:
                    source += WORDS[4]
                    if instance["explanation"] is not None:
                        target = instance["explanation"] + "\n" + WORDS[5] + " " + instance["label"]
        else:
            if self.shots == 0:
                source = passage + WORDS[0] + instance["question"] + "\n"
                if self.args.cot is None:
                    source += WORDS[8]
                else:
                    source += WORDS[9]
            else:
                source = WORDS[6] + passage + " " + instance["question"] + "\n"
                if self.args.cot is None:
                    source += WORDS[5]
                else:
                    source += WORDS[4]
                    if instance["explanation"] is not None:
                        target = instance["explanation"] + "\n" + WORDS[5] + " " + instance["label"]
        return dict(
            source=source,
            target=" " + target
        )

    def post_processing(self, predictions):
        new_predictions = []
        for pred in predictions:
            new_pred = self.post_process(pred)
            while new_pred.endswith("."):
                new_pred = new_pred[:len(new_pred) - 1]
            if self.task in AGIEVAL_NO_LETTER_CHOICE:
                new_pred = math_equiv._strip_string(new_pred)
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

    @property
    def references(self):
        if self.task not in AGIEVAL_NO_LETTER_CHOICE:
            return [[instance["label"].__str__()] for instance in self.evaluation_data]
        else:
            return [[math_equiv._strip_string(instance["label"].__str__())] for instance in self.evaluation_data]

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

    def remove_few_shot_prefix(self, string:str):
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
        if self.shots > 0 and self.args.cot is not None:
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

    def try_parse_few_shot_pattern(self, string:str):
        if self.shots > 0 and self.args.cot is not None:
            string = self.extract_last_line(string)
        if self.task in AGIEVAL_ZH_CLOZE:
            return string.startswith("答案是")
        elif self.task in AGIEVAL_EN_CLOZE:
            return string.startswith("The answer is therefore")
        elif self.task in AGIEVAL_ZH_QA:
            pattern = "答案是.*?([A-G])"
            match = re.search(pattern, string)
            return match is not None
        elif self.task in AGIEVAL_EN_QA:
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

    def extract_answer_in_bracket(answer, prefix='【', suffix='】'):
        if prefix not in answer and suffix not in answer:
            return ""
        s = answer.index(prefix) + len(prefix)
        t = answer.index(suffix)
        ret = answer[s:t]
        return ret

    def parse_math_answer(self, raw_string):
        if self.shots > 0 and self.args.cot is not None:
            raw_string = self.extract_last_line(raw_string)
        if self.shots > 0:
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

        def last_boxed_only_string(string):
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
            first_pattern = "\$(.*)\$"
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
                pattern = "(?:\\$)?\d+(?:\.\d+)?(?![\w\d])"
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
        if self.shots > 0 and self.args.cot is not None:
            string = self.extract_last_line(string)
        pattern = "\(*([A-F])\)*"
        match = re.findall(pattern, string)
        if match:
            answers = sorted(set(match))
            if self.task == "gaokao-physics" and len(answers) == 1:
                return answers[0]
            return answers.__str__()
        return ""

    def post_process(self, prediction):
        if self.task in AGIEVAL_NO_LETTER_CHOICE:
            return self.parse_math_answer(prediction)

        if self.task in AGIEVAL_MULTI_CHOICE:
            return self.parse_qa_multiple_answer(prediction)

        if self.shots == 0:
            answer = self._find_choice(prediction)
            return answer

        language = "en" if self.task in AGIEVAL_EN_QA else "zh"
        return self.parse_few_shot_qa_single_answer(prediction, language)
