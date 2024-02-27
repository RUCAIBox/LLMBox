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

    instruction = ""
    evaluation_set = "validation"
    example_set = "train"
    load_args = ("super_glue", "wsc")

    def format_instance(self, instance):

        def mark_word(sentence, index, word):
            words = sentence.split()
            words[index] = "*" + words[index][:len(word)] + "*" + words[index][len(word):]
            return " ".join(words)

        source = "Final Exam with Answer Key\nInstructions: Please carefully read the following passages. For each passage, you must identify which noun the pronoun marked in *bold* refers to.\n"
        source += "=====\n"
        if instance["idx"] != 42:
            modified_text = mark_word(instance["text"], instance["span2_index"], instance["span2_text"])
        else:
            modified_text = "When they had eventually calmed down a bit , and had gotten home, Mr. Farley put the magic pebble in an iron safe . Some day they might want to use *it* , but really for now, what more could they wish for"
        source += f"Passage: {modified_text}\n"
        source += f'Question: In the passage above, does the pronoun "*{instance["span2_text"]}*" refer to "{instance["span1_text"]}"?'
        options = [" No", " Yes"]
        return dict(
            source=source,
            source_postfix="\nAnswer:",
            target_idx=instance["label"],
            options=options,
        )

    @property
    def references(self):
        return [instance["label"] for instance in self.evaluation_data]
