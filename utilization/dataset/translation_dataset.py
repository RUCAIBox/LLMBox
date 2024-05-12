from functools import cached_property

from ..metric import Bleu
from .generation_dataset import GenerationDataset


class TranslationDataset(GenerationDataset):
    """The dataset of Wmt dataset.

    Example:
        subset_name: ro-en
        instance: {'translation': {'en': 'Obama welcomes Netanyahu', 'ro': 'Obama primește Netanyahu'}
        prediction: Obama receives Netanyahu
        reference: Obama welcomes Netanyahu
    """

    instruction = "Q: Translate to {{lang}}. {{translation[self.subset_name[:2]]}}\nA:"
    evaluation_set = "test"
    example_set = "train"
    metrics = [Bleu()]
    load_args = ()
    extra_model_args = dict(temperature=0, stop=["\n"])

    def init_arguments(self):
        from langcodes import Language
        self.language = Language(self.subset_name[3:5]).language_name("en")

    def format_instance(self, instance):
        instance["lang"] = self.language
        instance["target"] = instance[self.subset_name[3:5]]
        return instance

    @staticmethod
    def post_processing(preds):
        return [pred.strip().split("\n")[0] for pred in preds]

    @cached_property
    def references(self):
        return [instance["translation"][self.subset_name[3:]] for instance in self.evaluation_data]
