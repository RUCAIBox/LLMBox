from .multiple_choice_dataset import MultipleChoiceDataset


class Boolq(MultipleChoiceDataset):
    """The dataset of Boolq

    Boolean Questions (Boolq, Clark et al., 2019a) is a QA task where each example consists of a short passage and a yes/no question about the passage.

    Example:
        question: do iran and afghanistan speak the same language
        passage: It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958),
        label: 1
    """

    instruction = "{{passage}}\nquestion: {{question}}?{{'\n' + options if options}}\nanswer:"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("super_glue", "boolq")

    def format_instance(self, instance):
        instance["options"] = ["no", "yes"]
        return instance

    @property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
