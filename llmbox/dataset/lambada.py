from ..metric import Word_Accuracy
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
    load_args = ("EleutherAI/lambada_openai", "default")
    extra_model_args = dict(max_tokens=5, temperature=0)

    def __init__(self, args, model, subset_name=None):
        super().__init__(args, model, subset_name=subset_name)
        self.metrics = [Word_Accuracy(self.tokenizer)]

    def _format_instance(self, instance):
        """

        According to the README of the dataset, for dev and test set, the last word of the passage is the target, and the rest of the passage is the source.
        """
        instance["text"] = instance["text"].split()
        # get the last word of the passage
        target = " " + instance["text"][-1]
        # get the rest of the passage
        source = " ".join(instance["text"][:-1])
        return dict(source=source, target=target)

    @property
    def references(self):
        return [" " + instance["text"][-1] for instance in self.evaluation_data]
