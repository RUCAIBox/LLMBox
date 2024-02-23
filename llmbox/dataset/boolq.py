from .multiple_choice_dataset import MultipleChoiceDataset


class Boolq(MultipleChoiceDataset):
    """The dataset of Boolq

    Boolean Questions (Boolq, Clark et al., 2019a) is a QA task where each example consists of a short passage and a yes/no question about the passage.

    Example:
        question: do iran and afghanistan speak the same language
        passage: It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958),
        label: 1
    """

    instruction = ""
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("super_glue", "boolq")

    def _format_instance(self, instance):
        label2text = {
            0: " no",
            1: " yes",
        }

        options = [label2text[option] for option in [0, 1]]
        return dict(
            source=instance["passage"] + "\nquestion: " + instance["question"] + "?",
            source_postfix="\nanswer:",
            target_idx=instance["label"],
            options=options,
        )

    @property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
