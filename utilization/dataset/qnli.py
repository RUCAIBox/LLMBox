from functools import cached_property

from .multiple_choice_dataset import MultipleChoiceDataset


class Qnli(MultipleChoiceDataset):
    """The dataset of Qnli.

    The Stanford Question Answering Dataset is a question-answering dataset consisting of question-paragraph pairs, where one of the sentences in the paragraph (drawn from Wikipedia) contains the answer to the corresponding question (written by an annotator). The authors of the benchmark convert the task into sentence pair classification by forming a pair between each question and each sentence in the corresponding context, and filtering out pairs with low lexical overlap between the question and the context sentence. The task is to determine whether the context sentence contains the answer to the question. This modified version of the original task removes the requirement that the model select the exact answer, but also removes the simplifying assumptions that the answer is always present in the input and that lexical overlap is a reliable cue.

    Example:
        question: When did the third Digimon series begin?
        sentence: Unlike the two seasons before it and most of the seasons that followed, Digimon Tamers takes a darker and more realistic approach to its story featuring Digimon who do not reincarnate after their deaths and more complex character development in the original Japanese.
        label: 1
    """

    instruction = "Determine whether the context sentence contains the answer to the question or not.\nQuestion: {{question.strip()}}\nSentence: {{sentence.strip()}}{{'\n'+options if options}}\nAnswer: "
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("nyu-mll/glue", "qnli")

    def format_instance(self, instance):
        instance["options"] = ["yes", "no"]
        return instance

    @cached_property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
