from .generation_dataset import GenerationDataset
from datasets import load_dataset, load_from_disk
import evaluate


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

    def __init__(self, args, model):
        self.name = "math"
        dataset = load_dataset("lighteval/MATH")
        # dataset = load_from_disk("lighteval/MATH")
        self.example_data = list(dataset["train"])
        self.evaluation_data = list(dataset["test"])
        self.instruction = "Answer the following questions."

        self.metric = "accuracy"
        super().__init__(args, model)

    def extract_inner_content(self, text):
        # extract from \boxed{...}, where{} can be nested
        start = text.find("\\boxed{") + 7
        count = 1
        end = start
        while count > 0 and end < len(text):
            if text[end] == "{":
                count += 1
            elif text[end] == "}":
                count -= 1
            end += 1
        return text[start:end - 1]

    def answer_cleansing(self, preds):
        # TODO: 0 shot doesn't know to put the answer in '\boxed{...}'.
        predictions = []
        for pred in preds:
            match = self.extract_inner_content(pred)
            if match:
                predictions.append(match)
            else:
                predictions.append(pred)

        return predictions

    def format_instance(self, instance):
        instance["problem"] = "Q: " + instance["problem"] + "\n" + "A:"
        instance["solution"] = " " + instance["solution"]
        return dict(
            source=instance["problem"],
            target=instance["solution"],
        )

    def calculate_metric(self, predictions):
        predictions = self.answer_cleansing(predictions)
        exact_match = evaluate.load("exact_match")
        em_score = exact_match.compute(predictions=predictions, references=self.references, ignore_case=True, ignore_punctuation=True)["exact_match"]
        return {'Accuracy': em_score}

    @property
    def references(self):
        return [self.extract_inner_content(instance["solution"]) for instance in self.evaluation_data]