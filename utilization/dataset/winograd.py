from logging import getLogger

from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)


class Winograd(MultipleChoiceDataset):
    """The dataset of Winograd.

    A Winograd schema is a pair of sentences that differ in only one or two words and that contain an ambiguity that is resolved in opposite ways in the two sentences and requires the use of world knowledge and reasoning for its resolution

    Example:
        label : 0
        options: ['The city councilmen', 'The demonstrators']
        pronoun: 'they'
        pronoun_loc: 63
        text: 'The city councilmen refused the demonstrators a permit because they feared violence.'
    """

    instruction = "{source}"
    evaluation_set = "test"
    example_set = None
    load_args = ("winograd_wsc", "wsc273")  # specify subset from command line

    def format_instance(self, instance):
        pronoun_len = len(instance["pronoun"])
        question = instance["text"][:instance["pronoun_loc"]]
        completion = instance["text"][instance["pronoun_loc"] + pronoun_len:]
        contexts = [question.strip() + ' ' + option for option in instance["options"]]
        return dict(
            source=contexts,
            source_idx=instance["label"],
            target=completion,
        )

    @property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
