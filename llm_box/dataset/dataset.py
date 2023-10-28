import torch
import numpy as np
from typing import Set, Literal, Optional

from ...metric import MetricModule
from .utils import NotImplementedField, DataLoaderX


class Dataset(torch.utils.data.Dataset):
    r"""The base class object for all datasets.

    Args:
        args (Namespace): The global configurations.

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
        example_separator_string (str, *optional*): The string to separate each demonstration example.
    """

    metrics: Set[str] = NotImplementedField
    evaluation_type: Literal['ranking', 'generation'] = NotImplementedField

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.tokenizer = args.tokenizer
        self.instruction = args.instruction

        self.num_shots = args.num_shots
        self.max_example_tokens = args.max_example_tokens
        self.example_separator_string = args.example_separator_string
        self.examples = self.construct_examples()
        self.construct_instances()

        self.metric_modules = MetricModule(self.metrics)

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
        r"""Format one instance into source text and optinal all target texts (for ranking).

        Args:
            instance (Dict): an instance dict of multiple key-value pairs.

        Returns:
            Dict:
                ground_truth: Union[str, Tuple(str, str)]
                options (*optional*): List[Tuple(str, str)]
        """
        raise NotImplementedError(f"{self.name} dataset must implement the `format_instance` function.")

    def format_instruction_and_examples(self, instance):
        r"""Format one instance with the instruction and demonstration.

        Args:
            instance (Union[str, Tuple(str, str)]): a pre-formatted instance.

        Returns:
            Union[str, Tuple(str, str)]: The final formatted instance.
        """
        # TODO: instruction template
        # TODO: ICL
        if isinstance(instance, tuple):
            source, target = instance
            return self.instruction + self.examples + source, target
        else:
            return self.instruction + self.examples + instance

    def construct_examples(self, instance=None):
        r"""Format one instance with the instruction and demonstration.

        Args:
            instance (Union[str, Tuple(str, str)], *optional*): a pre-formatted evaluation instance.

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
            cur_example_text = self.format_instance(self.example_data[index])['ground_truth']
            if isinstance(cur_example_text, tuple):
                cur_example_text = ''.join(cur_example_text)
            cur_example_text += self.example_separator_string
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
            if self.evaluation_type == "ranking":
                options = [
                    self.format_instruction_and_examples(option) for option in self.format_instance(instance)['options']
                ]
                self.evaluation_instances.extend(options)
                self.option_nums.append(len(options))
            elif self.evaluation_type == "generation":
                self.evaluation_instances.append(
                    self.format_instruction_and_examples(self.format_instance(instance)['ground_truth'])
                )

    def calculate_metrics(self):
        r"""Calculate the metrics for the predictions.

        Args:
            predictions (List[str]): The list of predictions.

        Returns:
            Dict[str, float]: The metrics for the predictions.
        """
        return self.metric_modules.calculate_metric()

    def get_dataloader(self, **kwargs):
        default_kwargs = dict(
            batch_size=self.args.batch_size,
            collate_fn=lambda x: x,
            shuffle=False,
            pin_memory=True,
        )
        default_kwargs.update(kwargs)
        return DataLoaderX(self, **default_kwargs)

