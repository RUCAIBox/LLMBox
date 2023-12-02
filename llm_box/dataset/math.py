import re

import numpy as np

from .generation_dataset import GenerationDataset
from ..metric import Accuracy

SUBSTITUTIONS = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''), (' ', ''), ('mbox', 'text'),
                 (',\\text{and}', ','), ('\\text{and}', ','), ('\\text{m}', '\\text{}')]

REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft', 'hours', 'km', 'units', '\\ldots', 'sue', 'points',
    'feet', 'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals', 'edges', 'students',
    'childrentickets', 'multiples', '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}',
    '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots'
]


class Math(GenerationDataset):
    """The dataset of MATH.

    MATH(Hendrycks et al. 2021), a dataset of 12,500 challenging competition mathematics problems  with step-by-step solutions
    written in LATEX and natural language.

    Examples:
        problem: Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).
        level: Level 5
        type: Algebra
        solution: For the piecewise function to be continuous, the cases must "meet" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \Rightarrow a=-3$. Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\boxed{0}$.
    """

    name = "math"
    instruction = "Answer the following question."

    example_set = "train"
    evaluation_set = "test"

    load_args = ("hendrycks/competition_math",)
    metrics = [Accuracy()]

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
            if ('The answer is ' in pred):
                pred = pred.split('The answer is ')[-1].strip()
            final_answer = re.findall(pattern, pred)
            if final_answer:
                new_predictions.append(Math.normalize_final_answer(final_answer[-1]))
            else:
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
                new_predictions.append(numbers[-1] if numbers else pred)
        return new_predictions

    def format_instance(self, instance):
        instance["short_answer"] = self.extract_inner_content(instance["solution"])
        instance["problem"] = "Q: " + instance["problem"] + "\n" + "A:"
        instance["solution"] = " " + instance[
            "solution"] + f"\nFinal Answer: The final answer is ${instance['short_answer']}$. I hope it is correct."
        return dict(
            source=instance["problem"],
            target=instance["solution"],
        )

    @property
    def references(self):
        return [instance["short_answer"] for instance in self.evaluation_data]
