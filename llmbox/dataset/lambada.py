import re
import numpy as np

from ..metric import Accuracy
from .generation_dataset import GenerationDataset


class Lambada(GenerationDataset):
    r"""The dataset of lambada.

    The LAMBADA evaluates the capabilities of computational models for text understanding by means of a word prediction task. LAMBADA is a collection of narrative passages sharing the characteristic that human subjects are able to guess their last word if they are exposed to the whole passage, but not if they only see the last sentence preceding the target word. To succeed on LAMBADA, computational models cannot simply rely on local context, but must be able to keep track of information in the broader discourse.

    Example:
        'category': 'Mystery',
        'text': 'bob could have been called in at this point , but he was n't miffed at his exclusion at all . he was relieved at not being brought into this initial discussion with central command . `` let 's go make some grub , '' said bob as he turned to danny . danny did n't keep his stoic expression , but with a look of irritation got up and left the room with bob',

    """

    evaluation_set = "test"
    example_set = "validation"
    metrics = [
        Accuracy(),
    ]
    load_args = ("lambada",)
    extra_model_args = dict(max_tokens=3, n=1, temperature=0, best_of=4, length_penalty=0.6)

    def _format_source(self, source: str) -> str:
        """
        convert the source to the right format according to the paper
        """
        # using capital I, remove space before n't
        source = source.replace(" i ",
                                " I ").replace(" n't", "n't").replace(" 's", "'s").replace(" 're", "'re").replace(
                                    " 'm", "'m"
                                ).replace(" 've", "'ve").replace(" 'll", "'ll").replace(" 'd", "'d")
        # check if "''" appear before "``", if so, add "``" in the front
        if "''" in source:
            # make the start of the passage have the right format
            if "``" not in source:
                source = f"`` {source}"
            elif source.index("''") < source.index("``"):
                source = f"`` {source}"
        # remove extra space before "''" and extra space after "``"
        source = source.replace(" ''", "''").replace("`` ", "``")
        # detect if "``" appear right after "''", if so, insert a newline in between
        source = source.replace("'' ``", "''\n``")
        source = source.replace("``", "\"").replace("''", "\"")
        # remove extra space before punctuation
        source = source.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")

        # the first letter of each sentence should be capitalized
        # find the first of the whole passage and capitalize it
        source = source[0].upper() + source[1:] if source[0].isalpha() else source[0] + source[1].upper() + source[2:]
        # using regular expression to find the end position of each sentence
        end_positions = [m.end() for m in re.finditer(r'(," |(\.+|\?|!)"*( |\n))(?=("|[a-zA-Z]))', source)]
        for end_position in end_positions:
            # capitalize the first letter of the next sentence
            source = source[:end_position] + source[end_position].upper() + source[end_position + 1:]

        return source

    def format_instance(self, instance):
        """

        According to the README of the dataset, for dev and test set, the last word of the passage is the target, and the rest of the passage is the source.
        """

        # get the last word of the passage
        target = instance["text"].split()[-1]
        # get the rest of the passage
        source = " ".join(instance["text"].split()[:-1])

        source = self._format_source(source)

        return dict(source=source, target=target)

    @staticmethod
    def post_processing(predictions: list[str]):
        predictions = [pred.split() for pred in predictions]
        predictions = [pred[0] if pred else "" for pred in predictions]
        return [
            pred.replace(".", "").replace(",", "").replace('"', "").replace("?", "").replace("!", "").lower()
            for pred in predictions
        ]

    @property
    def references(self):
        return [instance["text"].split()[-1] for instance in self.evaluation_data]
