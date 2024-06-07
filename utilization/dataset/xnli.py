from functools import cached_property
from logging import getLogger

from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)


class Xnli(MultipleChoiceDataset):
    """The dataset of XNLI.

    XNLI (Conneau et al. 2018) is a subset of a few thousand examples from MNLI which has been translated into a 14 different languages (some low-ish resource).
    
    Example:
        "hypothesis": "Man verliert die Dinge auf die folgende Ebene , wenn sich die Leute erinnern .",
        "label": 0,
        "premise": "\"Du weißt , während der Saison und ich schätze , auf deiner Ebene verlierst du sie auf die nächste Ebene , wenn sie sich entschl..."
    """

    instruction = "Given the premise sentence in '{{lang}}': '{{premise.strip()}}', does the hypothesis sentence '{{hypothesis.strip()}}' entail, contradict, or neither (neutral) with respect to the premise?{{'\n'+options if options}}\nAnswer:"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("xnli",)
    banned_subsets = ["all_languages"]

    def init_arguments(self):
        from langcodes import Language
        self.language = Language(self.subset_name).language_name("en")

    def format_instance(self, instance):
        instance["lang"] = self.language
        instance["options"] = ["entailment", "neutral", "contradiction"]
        return instance

    @cached_property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
