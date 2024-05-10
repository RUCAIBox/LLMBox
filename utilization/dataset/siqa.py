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

    instruction = "{{context}}\nQuestion: {{question}}{{'\n' + options if options}}\nAnswer:"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("lighteval/siqa",)

    def format_instance(self, instance):
        instance["options"] = [instance["answerA"], instance["answerB"], instance["answerC"]]
        instance["target_idx"] = int(instance["label"]) - 1
        return instance

    @property
    def references(self):
        return [int(instance["label"]) - 1 for instance in self.evaluation_data]
