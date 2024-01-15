import re
import threading

from .generation_dataset import GenerationDataset
from ..metric import Accuracy


class Gsm8k(GenerationDataset):
    """The dataset of GSM8K.

    GSM8K(Cobbe et al. 2021), linguistically diverse grade school math word problems

    Examples:
        question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
        answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72
    """

    instruction = "Answer the following question."
    evaluation_set = "test"
    example_set = ""
    load_args = ("gsm8k", "main")
    metrics = [Accuracy()]
    model_args = dict(temperature=0)

    _decimal_separator = re.compile(r"(\d),(\d)")
    _extract_numbers = re.compile(r"[-+]?\d*\.\d+|\d+")

    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        super().load_raw_dataset(dataset_path, subset_name, evaluation_set, example_set)
        if self.args.cot == 'base':
            self.example_data = BASE_EXAMPLARS
        elif self.args.cot == 'least_to_most':
            self.example_data = LEAST_TO_MOST_EXAMPLARS
        elif self.args.cot == 'pal':
            self.example_data = PAL_EXAMPLARS
            self.instruction = "Let's use python to solve math problems. Here are some examples how to do it."
        
        if self.model.type == 'base':
            self.model_args['stop'] = ['\n']

    def post_processing(self, predictions):
        new_predictions = []
        for pred in predictions:
            if self.args.cot == 'pal':
                if '```python' in pred:
                    pred = pred.split('```python')[1].split('```')[0]
                elif '```' in pred:
                    pred = pred.split('```')[1].split('```')[0]
                code = pred.split('\n')

                with Timeout():
                    try:
                        exec('\n'.join(code))
                        ans = eval("solution()")
                        ans = str(ans)[:-2] if str(ans).endswith(".0") else str(ans)
                        new_predictions.append(ans)
                    except:
                        new_predictions.append('')
            else:
                # replace numbers like `x,xxx` with `xxxx`
                pred = self._decimal_separator.sub(r"\1\2", pred)
                numbers = self._extract_numbers.findall(pred)
                if numbers:
                    new_predictions.append(numbers[-1])
                else:
                    new_predictions.append(pred)
        return new_predictions

    def format_instance(self, instance):
        instance["question"] = instance["question"].replace("\n", " ")
        question = f'Question: {instance["question"]}\nAnswer:'

        instance["answer"] = ' ' + self._decimal_separator.sub(r"\1\2", instance["answer"])  # for example
        if "####" in instance["answer"]:
            instance['short_answer'] = instance["answer"].split("####")[1].strip()  # for reference

        return dict(
            source=question,
            target=instance["answer"],
        )

    @property
    def references(self):
        return [instance["short_answer"] for instance in self.evaluation_data]


class Timeout:

    def __init__(self, seconds=10, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
        self.timer = threading.Timer(self.seconds, self.timeout_handler)

    def timeout_handler(self):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        self.timer.start()

    def __exit__(self, type, value, traceback):
        self.timer.cancel()


BASE_EXAMPLARS = [{
    "question":
    "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
    "answer":
    "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6."
}, {
    "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
    "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5."
}, {
    "question":
    "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
    "answer":
    "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39."
}, {
    "question":
    "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
    "answer":
    "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8."
}, {
    "question":
    "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
    "answer":
    "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9."
}, {
    "question":
    "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
    "answer":
    "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29."
}, {
    "question":
    "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
    "answer":
    "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33."
}, {
    "question":
    "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
    "answer":
    "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8."
}]

LEAST_TO_MOST_EXAMPLARS = [{
    "question":
    "Elsa has 5 apples. Anna has 2 more apples than Elsa. How many apples do they have together?",
    "answer":
    """To answer the question "How many apples do they have together?", we need to know: "How many apples does Anna have?".
1. Anna has 2 more apples than Elsa. So Anna has 2 + 5 = 7 apples.
2. Elsa and Anna have 5 + 7 = 12 apples together. So the answer is 12."""
}, {
    "question":
    "If Pam is currently twice as young as Rena is, and in 10 years Rena will be 5 years older than her, how old is Pam now?",
    "answer":
    """To answer the question "How old is Pam now?", we need to know: "How much older is Rena than Pam currently?".
1. Since Rena will be 5 years older than Pam in 10 years, she must be 5 years older than Pam now as well.
2. If Pam is currently twice as young as Rena, that means that Rena is currently twice as old as Pam is. So if P stands for Pam's age now and R stands for Rena's age now, then we know that R = 2 * P And since Rena is 5 years older than Pam now, we know that R = P + 5. By substitution, we have P + 5 = 2 * P, which means that P = 5. So the answer is 5."""
}]

PAL_EXAMPLARS = [{
    "question":
    "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
    "answer":
    '''
```
def solution():
    """Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result
```'''
}, {
    "question":
    "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
    "answer":
    '''
```
def solution():
    """Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"""
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result
```'''
}, {
    "question":
    "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
    "answer":
    '''
```
def solution():
    """There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"""
    computers_initial = 9
    computers_per_day = 5
    num_days = 4  # 4 days between monday and thursday
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result
```'''
}, {
    "question":
    "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
    "answer":
    '''
```
def solution():
    """Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"""
    toys_initial = 5
    mom_toys = 2
    dad_toys = 2
    total_received = mom_toys + dad_toys
    total_toys = toys_initial + total_received
    result = total_toys
    return result
```'''
}, {
    "question":
    "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
    "answer":
    '''
```
def solution():
    """Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"""
    jason_lollipops_initial = 20
    jason_lollipops_after = 12
    denny_lollipops = jason_lollipops_initial - jason_lollipops_after
    result = denny_lollipops
    return result
```'''
}, {
    "question":
    "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
    "answer":
    '''
```
def solution():
    """Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"""
    leah_chocolates = 32
    sister_chocolates = 42
    total_chocolates = leah_chocolates + sister_chocolates
    chocolates_eaten = 35
    chocolates_left = total_chocolates - chocolates_eaten
    result = chocolates_left
    return result
```'''
}, {
    "question":
    "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
    "answer":
    '''
```
def solution():
    """If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"""
    cars_initial = 3
    cars_arrived = 2
    total_cars = cars_initial + cars_arrived
    result = total_cars
    return result
```'''
}, {
    "question":
    "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
    "answer":
    '''
```
def solution():
    """There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"""
    trees_initial = 15
    trees_after = 21
    trees_added = trees_after - trees_initial
    result = trees_added
    return result
```'''
}]
