import re

from ..metric import PassAtK
from .generation_dataset import GenerationDataset

class Mbpp(GenerationDataset):
    """The dataset of MBPP.

    The benchmark consists of around 1,000 crowd-sourced Python programming problems, designed to be solvable by entry level programmers, covering programming fundamentals, standard library functionality, and so on. Each problem consists of a task description, code solution and 3 automated test cases.

    Examples:
        task_id: 601
        text: Write a function to find the longest chain which can be formed from the given set of pairs.
        code: class Pair(object): def __init__(self, a, b): self.a = a self.b = b def max_chain_length(arr, n): max = 0 mcl = [1 for i in range(n)] for i in range(1, n): for j in range(0, i): if (arr[i].a > arr[j].b and mcl[i] < mcl[j] + 1): mcl[i] = mcl[j] + 1 for i in range(n): if (max < mcl[i]): max = mcl[i] return max
        test_list: [ "assert max_chain_length([Pair(5, 24), Pair(15, 25),Pair(27, 40), Pair(50, 60)], 4) == 3", "assert max_chain_length([Pair(1, 2), Pair(3, 4),Pair(5, 6), Pair(7, 8)], 4) == 4", "assert max_chain_length([Pair(19, 10), Pair(11, 12),Pair(13, 14), Pair(15, 16), Pair(31, 54)], 5) == 5" ]
        test_setup_code:
        challenge_test_list: []
    """

    instruction = ""
    example_set = "train"
    evaluation_set = "test"
    load_args = ("mbpp","full")
    extra_model_args = dict(temperature=0)

    def __init__(self, args, model, subset_name=None):
        super().__init__(args, model, subset_name=subset_name)
        self.metrics = [PassAtK(k=args.pass_at_k)]

    def format_instance(self, instance):
        prompt = instance["text"]
        tests = '\n'.join(instance["test_list"])
        code = instance["code"].replace("\r", "").replace("\t", "    ")
        source_prompt = f"You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{tests}\n"
        target_prompt = f"[BEGIN]\n{code}\n[DONE]\n"
        return dict(
            source=source_prompt,
            target=target_prompt
        )

    def post_processing(self, predictions):
        answer_pattern = re.compile(r"\[BEGIN\](.*?)\[DONE\]", re.DOTALL)
        return [re.search(answer_pattern, p).group(1).strip() if re.search(answer_pattern, p) else p for p in predictions]

    @property
    def references(self):
        return self.evaluation_data
