import torch
import numpy as np
from typing import Set, Literal, Optional

from torch.utils.data import DataLoader

from .utils import NotImplementedField


class Dataset(torch.utils.data.Dataset):
    r"""The base class object for all datasets.

    Args:
        args (Namespace): The global configurations.
        model (Model): Our class for model.

    Attributes:
        name (str): The name of this dataset.
        tokenizer (Union[transformers.PreTrainedTokenizer, tiktoken.Encoding]): The tokenizer of corresponding model.
        evaluation_data (List[dict]): The list of data for evaluation.
        evaluation_instances (List[Union[str, Tuple(str, str)]]): The list of formatted evaluation instances.
        evaluation_type (str): The method for evaluation, which can be set to either 'ranking' or 'generation'.
        metric (str): The metric for evaluating the predictions and references.
        instruction (str, *optional*): The instruction for this task.
        option_nums (List[int], *optional*): The list of the number of options for each instance (mainly used for multi-choice tasks).
        example_data (List[dict], *optional*): The list of demonstration data.
        num_shots (int, *optional*): The number of demonstration instances.
        max_example_tokens (int, *optional*): The number of maximum tokens in the demonstration.
    """
    name = ""
    metric = ""
    instruction = ""

    evaluation_type: Literal['ranking', 'generation'] = NotImplementedField

    def __init__(self, args, model):
        super().__init__()
        self.args = args

        self.model = model
        self.tokenizer = model.tokenizer

        self.num_shots = args.num_shots
        self.max_example_tokens = args.max_example_tokens
        self.examples = self.construct_examples()
        self.construct_instances()

    def __len__(self):
        return len(self.evaluation_instances)

    def __getitem__(self, idx):
        return self.evaluation_instances[idx]

    @property
    def references(self):
        r"""Get the references for `evaluation_data`.

        Returns:
            List[str]: The list of ground-truth answers.
        """
        raise NotImplementedError(f"{self.name} dataset must implement the `references` property.")

    def format_instance(self, instance):
        r"""Format the dataset instance into task source text, target text, and options (for ranking).

        Args:
            instance (Dict): an instance dict of multiple key-value pairs.

        Returns:
            Dict:
<<<<<<< HEAD
                source: str
                target: str
                options (*optional*): List[str]
=======
                ground_truth: Union[str, Tuple(str, str)]
                options (*optional*): List[Tuple(str, str)]
>>>>>>> cddbdb6 (Evaluator add support for datasets with multiple subsets, and add time performance)
        """
        raise NotImplementedError(f"{self.name} dataset must implement the `format_instance` function.")

    def format_instruction_and_examples(self, source, target=""):
        r"""Format one instance with the instruction and demonstration.

        Args:
            source (str): the pre-formatted source text.
            target (str, *optional*): the pre-formatted target text (default to "").

        Returns:
            Union[str, Tuple(str, str)]: The final formatted instance.
        """
        # TODO: instruction template
        # TODO: ICL

        if self.model.type == 'base':
            source = self.examples + source
        elif self.model.type == 'instruction':
            source = self.instruction + "\n\n" + self.examples + source

        if target:
            return source, target
        else:
            return source

    def construct_examples(self, instance=None):
        r"""Format one instance with the instruction and demonstration.

        Args:
            instance (Dict): a pre-formatted evaluation instance.

        Returns:
            str: The constructed demonstration text.
        """
        # selection algorithm
        # TODO: ICL
        indice = np.random.choice(len(self.example_data), self.args.num_shots)

        # TODO: tokenizer efficiency
        # construct few-shot examples
        example_text = ""
        example_token_nums = 0
        for index in indice:
            example = self.format_instance(self.example_data[index])
            cur_example_text = self.args.instance_format.format_map(example) + "\n\n"
            cur_token_num = len(self.tokenizer.encode(cur_example_text))
            if cur_token_num + example_token_nums <= self.max_example_tokens:
                example_text += cur_example_text
                example_token_nums += cur_token_num
        return example_text

    def construct_instances(self):
        r"""Construct and format all the instances of `evaluation_data`.

        Returns:
            List[str]: The list of final formatted instances.
        """
        self.evaluation_instances = []
        self.option_nums = []
        for instance in self.evaluation_data:
            formatted_instance = self.format_instance(instance)
            if self.evaluation_type == "ranking":
                options = [
                    self.format_instruction_and_examples(formatted_instance["source"], option)
                    for option in formatted_instance['options']
                ]
                self.evaluation_instances.extend(options)
                self.option_nums.append(len(options))
            elif self.evaluation_type == "generation":
                self.evaluation_instances.append(self.format_instruction_and_examples(formatted_instance["source"]))

    def calculate_metrics(self):
        r"""Calculate the metrics for the predictions.

        Args:
            predictions (List[str]): The list of predictions.

        Returns:
            Dict[str, float]: The metrics for the predictions.
        """
        raise NotImplementedError(f"{self.name} dataset must implement the `calcuate_metric` function.")

    def get_dataloader(self, **kwargs):
        default_kwargs = dict(
            batch_size=self.args.batch_size,
            collate_fn=lambda x: x,
            shuffle=False,
            pin_memory=True,
        )
        default_kwargs.update(kwargs)
        return DataLoader(self, **default_kwargs)
