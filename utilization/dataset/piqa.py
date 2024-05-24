from functools import cached_property

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

    instruction = "Question: {{goal}}{{'\n' + options if options}}\nAnswer:"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("piqa",)

    def format_instance(self, instance):
        instance["options"] = [instance['sol1'], instance['sol2']]
        return instance

    @cached_property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
