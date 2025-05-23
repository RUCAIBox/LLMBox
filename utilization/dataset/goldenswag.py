import re
from functools import cached_property

from .multiple_choice_dataset import MultipleChoiceDataset


class Goldenswag(MultipleChoiceDataset):
    """The dataset of golenswag. http://arxiv.org/abs/2504.07825

    HellaSwag: Can a Machine Really Finish Your Sentence? (Zellers et al., 2019)
    Hellaswag is a new dataset for commonsense NLI. The paper was published at ACL2019.

    Example:
        'activity_label': 'Roof shingle removal',
        'ctx_a': 'A man is sitting on a roof.',
        'ctx_b': 'he',
        'ctx': 'A man is sitting on a roof. he',
        'endings': ['is using wrap to wrap a pair of skis.',
                    'is ripping level tiles off.',
                    "is holding a rubik's cube.",
                    'starts pulling up roofing on a roof.'],
        'label': '3'
    """

    instruction = "{{ preprocess(activity_label + ': ' + ctx_a + ' ' + ctx_b.capitalize()) }}{{'\n' + options + '\nAnswer:' if options}}"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("PleIAs/GoldenSwag",)

    def init_arguments(self):
        self.jinja2_env.globals["preprocess"] = self.preprocess

    @staticmethod
    def preprocess(text):
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text.strip()

    def format_instance(self, instance):
        instance["options"] = [self.preprocess(instance["endings"][i]) for i in range(4)]
        return instance

    @cached_property
    def references(self):
        return [int(instance["label"]) for instance in self.evaluation_data]
