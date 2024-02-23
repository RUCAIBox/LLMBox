from .multiple_choice_dataset import MultipleChoiceDataset


class PIQA(MultipleChoiceDataset):
    """The dataset of PIQA.

    The PIQA dataset introduces the task of physical commonsense reasoning and a corresponding benchmark dataset Physical Interaction: Question Answering or PIQA.

    Example:
        'goal': "How do I ready a guinea pig cage for it's new occupants?",
        'label': 0,
        'sol1': 'Provide the guinea pig with a cage full of a few inches of bedding '
                'made of ripped paper strips, you will also need to supply it with a '
                'water bottle and a food dish.',
        'sol2': 'Provide the guinea pig with a cage full of a few inches of bedding '
                'made of ripped jeans material, you will also need to supply it with '
                'a water bottle and a food dish.'
    """

    instruction = ""
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("piqa",)

    def _format_instance(self, instance):
        source_text = "Question: " + instance['goal']
        options = [' ' + instance[option] for option in ['sol1', 'sol2']]
        return dict(
            source=source_text,
            source_postfix="\nAnswer:",
            target_idx=instance["label"],
            options=options,
        )

    @property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
