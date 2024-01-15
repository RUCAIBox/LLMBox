from .multiple_choice_dataset import MultipleChoiceDataset


class Wsc(MultipleChoiceDataset):
    """The dataset of Wsc

    Winograd Schema Challenge (Wsc, Levesque et al., 2012) is a coreference resolution task in which examples consist of a sentence with a pronoun and a list of noun phrases from the sentence.

    Example:
        text: Mark told Pete many lies about himself, which Pete included in his book. He should have been more skeptical.
        span1_index: 0
        span2_index: 13
        span1_text: Mark
        span2_text: He
        label: 0
    """
    instruction = "Final Exam with Answer Key"
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("super_glue", "wsc")

    def format_instance(self, instance):

        def mark_word(sentence, index):
            words = sentence.split()
            if index < 0 or index >= len(words):
                return "OverIndex"
            front = 0
            rear = len(words[index])
            while str.isalpha(words[index][front]) == False and front < rear:
                front += 1
            while str.isalpha(words[index][rear - 1]) == False and front < rear:
                rear -= 1
            if front == rear:
                print("Not word")
                return sentence
            words[index] = (words[index][:front] + "*" + words[index][front:rear] + "*" + words[index][rear:])
            return " ".join(words)

        source = "Instructions: Please carefully read the following passages. For each passage, you must identify which noun the pronoun marked in *bold* refers to.\n"
        source += '=====\n'
        modified_text = mark_word(instance['text'], instance['span2_index'])
        source += f'Passage: {modified_text}\n'
        source += f'Question: In the passage above, does the pronoun "*{instance["span2_text"]}*" refer to "{instance["span1_text"]}"?\n'
        source += "Answer:"
        label2text = {
            0: ' no',
            1: ' yes',
        }

        options = [label2text[option] for option in [0, 1]]
        return dict(
            source=source,
            target=label2text[instance['label']],
            options=options,
        )

    @property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
