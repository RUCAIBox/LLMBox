import re
from functools import cached_property
from logging import getLogger

from ..dataset_enum import AGIEVAL_SUBJECTS, AGIEVAL_ZH_PROMPT_TASKS
from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)

uncleaned_label = re.compile(r"^(\([ABCDEFGHIJ]\)|[ABCDEFGHIJ] *\.) *")

INSTRUCTIONS = {
    "zh_zero_shot": "{passage}问题：{question}\n{options}答案：从A到{max_option_letter}，我们应选择",
    "zh_few_shot": "问题. {passage} {question}\n{options}从以下选项中选择：",
    "en_zero_shot": "{passage}Q: {question}\n{options}Answer: Among A through {max_option_letter}, the answer is",
    "en_few_shot": "Question. {passage} {question}\n{options}Choose from the following options: ",
}


class Agieval_single_choice(MultipleChoiceDataset):
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
    load_args = ("RUCAIBox/agieval-single-choice",)
    categorized_subsets = AGIEVAL_SUBJECTS

    def init_arguments(self):
        # FIXME
        self.prefix_caching = False

        text = ""
        text += "zh" if self.subset_name in AGIEVAL_ZH_PROMPT_TASKS else "en"
        text += "_few_shot" if self.max_num_shots > 0 else "_zero_shot"
        self.instruction = INSTRUCTIONS[text]

    def format_instance(self, instance):
        instance["max_option_letter"] = self._max_choice_letter(instance["options"])
        instance["target_idx"] = int(ord(instance["label"].strip()[0]) - ord('A'))
        # remove reduntant label prefix (e.g. "(A) xxx" -> "xxx")
        instance["options"] = list(map(lambda op: uncleaned_label.split(op.strip())[-1], instance["options"]))
        return instance

    def calculate_metric(self, predictions):
        results, score_lists = super().calculate_metric(predictions)
        return results, score_lists

    @staticmethod
    def _max_choice_letter(choices):
        return chr(ord('A') + len(choices) - 1)

    @cached_property
    def references(self):
        return [
            ord(instance["label"].strip()[0]) - ord('A')  # type: ignore
            for instance in self.evaluation_data
        ]
