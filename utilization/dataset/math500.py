import re
from functools import cached_property
from typing import Optional

from ..metric import Accuracy
from .dataset_utils import get_raw_dataset_loader
from .generation_dataset import GenerationDataset
from .math import K0_MATH_EXAMPLARS

SUBSTITUTIONS = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''), (' ', ''), ('mbox', 'text'),
                 (',\\text{and}', ','), ('\\text{and}', ','), ('\\text{m}', '\\text{}')]

REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft', 'hours', 'km', 'units', '\\ldots', 'sue', 'points',
    'feet', 'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals', 'edges', 'students',
    'childrentickets', 'multiples', '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}',
    '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots'
]


class Math500(GenerationDataset):
    r"""This dataset contains a subset of 500 problems from the MATH benchmark that OpenAI created in their Let's Verify Step by Step paper.

    Examples:
        problem: Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).
        level: Level 5
        type: Algebra
        solution: For the piecewise function to be continuous, the cases must "meet" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \Rightarrow a=-3$. Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\boxed{0}$.
    """

    instruction = "Solve the following math problem.\n\nQuestion: {problem}\nAnswer:"
    target_template = "{solution}\nFinal Answer: The final answer is ${short_answer}$. I hope it is correct."
    example_set = "train"
    evaluation_set = "test"
    load_args = ("HuggingFaceH4/MATH-500",)
    metrics = [Accuracy()]
    extra_model_args = dict(temperature=0)
    supported_cot = ["k0_math"]

    def load_raw_dataset(self, dataset_path: Optional[str], subset_name: Optional[str], evaluation_set: str, example_set: Optional[str]):
        super().load_raw_dataset(dataset_path, subset_name, evaluation_set, None)
        if self.cot == 'k0_math':
            self.example_data = K0_MATH_EXAMPLARS
        else:
            load_fn = get_raw_dataset_loader(
                dataset_name="math",
                dataset_path=None,
                subset_name=subset_name,
                load_args=("hendrycks/competition_math",),
                load_kwargs=None,
            )
            self.example_data = load_fn(self.example_set)

    def init_arguments(self):
        if self.model_type == 'base':
            # when evaluating base model, responses might be in multiple lines
            self.extra_model_args.get("stop", []).append("\n\n")

    @staticmethod
    def normalize_final_answer(final_answer: str) -> str:
        """Normalize a final answer to a quantitative reasoning question."""
        final_answer = final_answer.split('=')[-1]

        for before, after in SUBSTITUTIONS:
            final_answer = final_answer.replace(before, after)
        for expr in REMOVED_EXPRESSIONS:
            final_answer = final_answer.replace(expr, '')

        # Extract answer that is in LaTeX math, is bold,
        # is surrounded by a box, etc.
        final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
        final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

        # Normalize shorthand TeX:
        # \fracab -> \frac{a}{b}
        # \frac{abc}{bef} -> \frac{abc}{bef}
        # \fracabc -> \frac{a}{b}c
        # \sqrta -> \sqrt{a}
        # \sqrtab -> sqrt{a}b
        final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
        final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
        final_answer = final_answer.replace('$', '')

        # Normalize 100,000 -> 100000
        if final_answer.replace(',', '').isdigit():
            final_answer = final_answer.replace(',', '')

        return final_answer

    @staticmethod
    def extract_inner_content(text):
        # extract from \boxed{...}, where{} can be nested
        start = text.find("\\boxed{")
        if start == -1:
            return None
        start += 7
        count = 1
        end = start
        while count > 0 and end < len(text):
            if text[end] == "{":
                count += 1
            elif text[end] == "}":
                count -= 1
            end += 1
        return text[start:end - 1]

    @staticmethod
    def post_processing(predictions):
        new_predictions = []
        pattern = r'\$(.*?)\$'
        for pred in predictions:
            if ('final answer' in pred):
                pred = pred.split('The answer is ')[-1].strip()
            final_answer = re.findall(pattern, pred)
            if final_answer:
                new_predictions.append(Math500.normalize_final_answer(final_answer[-1]))
            else:
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
                new_predictions.append(numbers[-1] if numbers else pred)
        return new_predictions

    def format_instance(self, instance):
        instance["short_answer"] = self.extract_inner_content(instance["solution"])
        instance["target"] = self.target_template.format_map(instance)
        return instance

    @cached_property
    def references(self):
        return [instance["short_answer"] for instance in self.evaluation_data]
