from .multiple_choice_dataset import MultipleChoiceDataset


class PIQA(MultipleChoiceDataset):
    """The dataset of WinoGrande.

    WinoGrande is a new collection of 44k problems, inspired by Winograd Schema Challenge
    (Levesque, Davis, and Morgenstern 2011), but adjusted to improve the scale and robustness against the
    dataset-specific bias. Formulated as a fill-in-a-blank task with binary options, the goal is to choose the right
    option for a given sentence which requires commonsense reasoning.

    Each question is associated with 4 candidate answers, one of which is correct.

    Example:
        article:
        The rain had continued for a week and the flood had created a big river which were ... with tears.

        question: What did Nancy try to do before she fell over?

        answer: C

        options':
        [
        'Measure the depth of the river',
        'Look for a fallen tree trunk',
        'Protect her cows from being drowned',
        'Run away from the flooded farm'
        ]
    """

    instruction = ""
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("piqa", )  # specify subset from command line

    def format_instance(self, instance):
        source_text = "Question:\n" + instance['goal'] + '\n' + 'Answer:\n'
        options_feat = ['sol1', 'sol2']
        options = [instance[option] for option in options_feat]
        return dict(
            source=source_text,
            target=options[instance["label"]],
            options=options,
        )

    @property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
