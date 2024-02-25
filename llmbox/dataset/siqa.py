from .multiple_choice_dataset import MultipleChoiceDataset


class Siqa(MultipleChoiceDataset):
    """The dataset of SIQA

    The Social Interaction QA (SIQA,Maarten Sap,2019) dataset is a benchmark designed for assessing social commonsense intelligence in question-answering systems.

    Example:
        context: Jan needed to give out jobs for an upcoming project at work.
        question: What will Others want to do next?
        answerA: disagree with Jan
        answerB: get to work
        answerC: argue with the assignments
        label: 2
    """

    instruction = ""
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("lighteval/siqa",)

    def format_instance(self, instance):
        source = instance["context"] + "\nQuestion: " + instance["question"]

        label2text = {
            "1": " " + instance["answerA"],
            "2": " " + instance["answerB"],
            "3": " " + instance["answerC"],
        }

        options = [label2text[option] for option in ["1", "2", "3"]]
        return dict(
            source=source,
            source_postfix="\nAnswer:",
            target_idx=int(instance["label"]) - 1,
            options=options,
        )

    @property
    def references(self):
        return [int(instance["label"]) - 1 for instance in self.evaluation_data]
