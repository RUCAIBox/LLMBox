from ..metric import PPL
from .validation_perplexity_dataset import ValidationPerplexityDataset


class LambadaPPL(ValidationPerplexityDataset):
    r"""The dataset of lambada ppl.

    The LAMBADA evaluates the capabilities of computational models for text understanding by means of a word prediction task. LAMBADA is a collection of narrative passages sharing the characteristic that human subjects are able to guess their last word if they are exposed to the whole passage, but not if they only see the last sentence preceding the target word. To succeed on LAMBADA, computational models cannot simply rely on local context, but must be able to keep track of information in the broader discourse.

    Example:
        'category': 'Mystery',
        'text': 'bob could have been called in at this point , but he was n't miffed at his exclusion at all . he was relieved at not being brought into this initial discussion with central command . `` let 's go make some grub , '' said bob as he turned to danny . danny did n't keep his stoic expression , but with a look of irritation got up and left the room with bob',

    """

    evaluation_set = "test"
    load_args = ("EleutherAI/lambada_openai", "default")
    metrics = [PPL()]

    def format_instance(self, instance):
        return {
            "text": instance["text"],
            "options": [instance["text"]],
        }

