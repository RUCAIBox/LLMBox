from functools import cached_property
from logging import getLogger

from .multiple_choice_dataset import MultipleChoiceDataset

logger = getLogger(__name__)


class ImbueCode(MultipleChoiceDataset):
    """The dataset of Imbue code understanding questions.

    These examples fall into 2 categories:
    - "cloze": fill in the hole to produce the specified outcome;
    - "eval": given a snippet of python code, determine the outcome.
    Some questions are very easy, some are much more challenging. Most (if not all) of these questions should be relatively straightforward for an experienced programmer, even without a pencil and paper. Released as part of Imbue's 70b evals post.

    Link: https://huggingface.co/datasets/imbue/code-comprehension?row=0

    Example (To avoid data contamination, some fields are omitted):
        'question': 'If we execute the code below, what will `result` be equal to? ```python ... ```'
        'choices': [ "'66-66-66-foo'", "'foo-66-66-66'", "'66--66--66--foo'", "''" ]
        'correct_answer': '66- ... -foo'
    """

    instruction = "{{question}}{{'\n' + options if options}}\nAnswer:"
    evaluation_set = "train"
    example_set = None
    load_args = ("imbue/code-comprehension", )

    def format_instance(self, instance):
        instance["target_idx"] = instance["choices"].index(
            instance["correct_answer"])
        instance["options"] = instance["choices"]
        return instance

    @cached_property
    def references(self):
        return [instance["target_idx"] for instance in self.evaluation_data]
