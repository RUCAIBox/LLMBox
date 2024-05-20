from functools import cached_property

from .multiple_choice_dataset import MultipleChoiceDataset


class Mrpc(MultipleChoiceDataset):
    """The dataset of Mrpc.

    The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005) is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.

    Example:
        sentence1: Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .
        sentence2: Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .
        label: 1
    """

    instruction = "Determine whether the following 2 sentences are semantically equivalent.\n\nsentence1: {{sentence1.strip()}}\nsentenc2: {{sentence2.strip()}}{{'\n'+options if options}}\nAnswer: "
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("nyu-mll/glue", "mrpc")

    def format_instance(self, instance):
        instance["options"] = ["not_equivalent", "equivalent"]
        return instance

    @cached_property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
