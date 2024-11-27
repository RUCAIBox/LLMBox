import re
import signal
from functools import cached_property

from ..metric import Accuracy
from .generation_dataset import GenerationDataset

BASE_INSTRUCTION = "Answer the following question.\n\nQuestion: {{question.replace('\n', ' ')}}\nAnswer:"
PAL_INSTRUCTION = "Let's use python to solve math problems. Here are some examples how to do it.\n\nQuestion: {{question.replace('\n', ' ')}}\nAnswer:"


class Gsm8k(GenerationDataset):
    """The dataset of GSM8K.

    GSM8K(Cobbe et al. 2021), linguistically diverse grade school math word problems

    Examples:
        question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
        answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72
    """

    instruction = BASE_INSTRUCTION
    evaluation_set = "test"
    example_set = "train"
    load_args = ("gsm8k", "main")
    metrics = [Accuracy()]
    extra_model_args = dict(temperature=0)
    supported_cot = ["base", "least_to_most", "pal", "k0_math"]

    _decimal_separator = re.compile(r"(?<=\d),(?=\d)")
    _extract_numbers = re.compile(r"[-+]?\d*\.\d+|\d+")

    def init_arguments(self):
        if self.cot is None:
            # when using chain-of-thought, responses might be in multiple lines
            self.extra_model_args["stop"] = ["\n"]

    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        super().load_raw_dataset(dataset_path, subset_name, evaluation_set, example_set)
        if self.cot == 'base' or self.cot is None:
            self.example_data = BASE_EXAMPLARS
        elif self.cot == 'least_to_most':
            self.example_data = LEAST_TO_MOST_EXAMPLARS
        elif self.cot == 'pal':
            self.example_data = PAL_EXAMPLARS
            self.instruction = PAL_INSTRUCTION
        elif self.cot == 'k0_math':
            self.example_data = K0_MATH_EXAMPLARS

    def post_processing(self, predictions):
        new_predictions = []
        for pred in predictions:
            if self.cot == 'pal':
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
                # matches teh decimal separators like x,xxx,xxx
                pred = self._decimal_separator.sub("", pred)
                numbers = self._extract_numbers.findall(pred)
                if numbers:
                    # remove trailing zeros
                    number = re.sub(r"(?<=\d)\.0*$", "", numbers[-1])
                    new_predictions.append(number)
                else:
                    new_predictions.append(pred)
        return new_predictions

    def format_instance(self, instance):

        # remove decimal seperators
        instance["answer"] = self._decimal_separator.sub("", instance["answer"])

        # few-shot examples might not contain "####"
        if "####" in instance["answer"]:
            instance['short_answer'] = instance["answer"].split("####")[1].strip()  # for reference

        instance["target"] = instance["answer"]  # for few-shots example
        return instance

    @cached_property
    def references(self):
        return [instance["short_answer"] for instance in self.evaluation_data]


class Timeout:

    def __init__(self, seconds=10, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


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

K0_MATH_EXAMPLARS = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "Okay, let's see. I have this problem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n\nAlright, so initially, there are 15 trees. After planting, there are 21 trees. I need to find out how many trees were planted.\n\nFirst thought: Maybe I can subtract the original number from the final number to find out the difference.\n\nSo, let's do that. \\(21 - 15\\). Hmm, \\(21 - 15 = 6\\). So, did they plant 6 trees?\n\nWait, let me make sure that's correct. Maybe I should double-check my subtraction.\n\nLet me count: Starting from 15, adding 6 would give me... 15 plus 6 is 21. Yeah, that matches the final number of trees.\n\nBut just to be thorough, maybe I can think of it another way. Suppose the workers planted \\(x\\) trees today. Then the total number of trees after planting would be \\(15 + x = 21\\).\n\nSo, to find \\(x\\), I can solve the equation \\(15 + x = 21\\).\n\nSubtracting 15 from both sides: \\(x = 21 - 15\\). Oh, that's the same thing I did earlier. \\(x = 6\\).\n\nOkay, so both methods give me the same answer. It seems like the workers planted 6 trees today.\n\nJust to be extra sure, let's visualize it. If I have 15 trees and I add 6 more, do I get 21?\n\nLet me draw it out:\n\n- Starting with 15 trees: \\( \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\)\n- Adding 6 trees: \\( \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\)\n- Total: \\( \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\bullet \\)\n\nYep, that's 21 trees in total.\n\nSo, I'm confident now that the grove workers planted \\(\\boxed{6}\\) trees today.",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "Alright, let's tackle this problem. So, there are initially 3 cars in the parking lot, and then 2 more cars arrive. I need to find out the total number of cars now.\n\nFirst, I think this is a simple addition problem. But to make sure I understand it fully, let me break it down.\n\nI start by visualizing the parking lot. There are 3 cars already there. Then, 2 more cars come in. So, I have to add these two quantities together.\n\nI recall that addition is the operation of combining two numbers to find their total. So, I set up the equation:\n\n\\[ 3 + 2 \\]\n\nNow, let's compute this. 3 plus 2 is... hmm, let me think. I know that 2 plus 3 is 5, and addition is commutative, so the order doesn't matter. Therefore, 3 plus 2 should also be 5.\n\nBut wait, to be sure, let me use another method to verify. Maybe I can count it out.\n\nI have 3 cars initially:\n1. Car 1\n2. Car 2\n3. Car 3\n\nThen, 2 more cars arrive:\n4. Car 4\n5. Car 5\n\nCounting them all together: 1, 2, 3, 4, 5. That makes 5 cars in total.\n\nJust to double-check, I'll use another strategy—number lines. Starting at 3, I'll add 2 by moving two steps to the right:\n\n3 → 4 → 5\n\nYes, that confirms my initial calculation.\n\nHowever, let me consider if there's another way to approach this, just to ensure I'm not missing anything. Maybe using subtraction? If I have 5 cars now and I remove the 2 that arrived, I should get back to the original 3.\n\n\\[ 5 - 2 = 3 \\]\n\nThat checks out.\n\nAlternatively, I can think of it in terms of multiplication. If I have 2 groups of cars: one group with 3 cars and another with 2 cars, the total is:\n\n\\[ 3 + 2 = 5 \\]\n\nStill, it's an addition problem, so multiplication isn't directly applicable here.\n\nI also thought about using fractions, but that seems unnecessary for this straightforward addition problem.\n\nFeeling confident now, I'll state the final answer clearly:\n\n\\[ \\boxed{5} \\]",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Okay, let's tackle this problem. Leah has 32 chocolates, and her sister has 42. Together, that's... hmm, let me add those up. 32 plus 42. I think it's 74. Yeah, 30 plus 40 is 70, and 2 plus 2 is 4, so 70 plus 4 is 74. So together, they have 74 chocolates.\n\nNow, they ate 35 chocolates. So I need to subtract 35 from 74 to find out how many are left. Let me do that: 74 minus 35.\n\nWait, 74 minus 30 is 44, and then minus 5 more is 39. So, 74 minus 35 equals 39.\n\nBut just to make sure I didn't mess up, I'll check it another way. Let's break it down:\n\\[ 74 - 35 = (70 + 4) - (30 + 5) = (70 - 30) + (4 - 5) = 40 - 1 = 39 \\]\nYeah, that confirms it.\n\nSo, after eating 35 chocolates, they have \\(\\boxed{39}\\) chocolates left in total.",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Alright, let's tackle this problem. Jason had 20 lollipops, gave some to Denny, and now has 12 left. I need to find out how many he gave to Denny.\n\nOkay, so first, I know that the total number of lollipops Jason started with was 20. He gave away some, and now he has 12. So, the number he gave away should be the difference between 20 and 12.\n\nLet me write that down:\n\n\\[ \\text{Lollipops given to Denny} = 20 - 12 \\]\n\nCalculating that:\n\n\\[ 20 - 12 = 8 \\]\n\nWait, is that right? Let me double-check. If Jason had 20 and gave away 8, then he would have:\n\n\\[ 20 - 8 = 12 \\]\n\nYes, that matches the information given in the problem. So, Jason gave Denny 8 lollipops.\n\n\\[ \\boxed{8} \\]",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": 'Alright, let\'s tackle this problem. Shawn has five toys, and he received two toys each from his mom and dad for Christmas. I need to find out how many toys he has now.\n\nOkay, first, I\'ll try to understand what\'s being asked. Shawn originally has five toys. Then, he gets two toys from his mom and two toys from his dad. So, he receives a total of \\(2 + 2 = 4\\) toys as gifts. I need to add these 4 toys to his original 5 toys.\n\nLet me write that down:\n\n\\[ \\text{Original toys} = 5 \\]\n\\[ \\text{Toys from mom} = 2 \\]\n\\[ \\text{Toys from dad} = 2 \\]\n\\[ \\text{Total toys received} = 2 + 2 = 4 \\]\n\nNow, adding the received toys to the original number:\n\n\\[ \\text{Total toys now} = 5 + 4 = 9 \\]\n\nWait, that seems straightforward. But let me double-check to make sure I didn\'t miss anything.\n\nMaybe I should consider if the toys from his mom and dad are included in the original five toys or not. If the original five toys already include the gifts from his parents, then the total would still be five. But the problem says he "received" two toys each from his mom and dad, which implies these are additional toys.\n\nSo, my initial calculation seems correct:\n\n\\[ 5 \\ (\\text{original toys}) + 2 \\ (\\text{from mom}) + 2 \\ (\\text{from dad}) = 9 \\]\n\nTo verify, let\'s think about it differently. Suppose Shawn has 5 toys initially. He gets 2 toys from his mom, so now he has:\n\n\\[ 5 + 2 = 7 \\]\n\nThen, he gets 2 toys from his dad:\n\n\\[ 7 + 2 = 9 \\]\n\nYes, that matches my previous result. So, Shawn now has \\(\\boxed{9}\\) toys.'
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "Alright, let's tackle this problem. Okay, so there were nine computers in the server room initially. Then, five more computers were installed each day from Monday to Thursday. I need to find out how many computers are there now.\n\nFirst, I need to figure out how many days there are from Monday to Thursday. Let's see, Monday, Tuesday, Wednesday, Thursday. That's four days. So, computers were added for four days.\n\nNow, each day five computers were installed. So, over four days, that would be \\(5 \\times 4 = 20\\) computers added in total.\n\nWait, hold on. Was it five computers each day, or five total per day? No, it says five more computers were installed each day. So, five per day.\n\nSo, total added computers: \\(5 \\times 4 = 20\\).\n\nNow, adding that to the initial number of computers: \\(9 + 20 = 29\\).\n\nLet me double-check that. Starting with nine computers:\n\n- Monday: 9 + 5 = 14\n- Tuesday: 14 + 5 = 19\n- Wednesday: 19 + 5 = 24\n- Thursday: 24 + 5 = 29\n\nYep, that adds up to 29 computers.\n\nSo, the total number of computers now in the server room is \\(\\boxed{29}\\)."
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Alright, let's tackle this problem. Michael had 58 golf balls initially. On Tuesday, he lost 23, and then on Wednesday, he lost 2 more. I need to find out how many golf balls he has at the end of Wednesday.\n\nFirst, I'll start by understanding the problem. Michael starts with 58 golf balls. He loses some on Tuesday and then more on Wednesday. So, the total number of golf balls he loses is 23 + 2 = 25. If I subtract the total lost from the initial amount, I should get the remaining number of golf balls.\n\nBut wait, let me double-check that approach. Maybe I should consider each day separately to avoid mistakes.\n\nOn Tuesday, Michael loses 23 golf balls. So, after Tuesday, the number of golf balls he has is:\n\\[ 58 - 23 = 35 \\]\n\nThen, on Wednesday, he loses 2 more golf balls. So, after Wednesday, the number of golf balls he has is:\n\\[ 35 - 2 = 33 \\]\n\nTherefore, at the end of Wednesday, Michael has \\(\\boxed{33}\\) golf balls.\n\nLet me verify this to be sure. Starting with 58, losing 23 on Tuesday:\n\\[ 58 - 23 = 35 \\]\nThen losing 2 on Wednesday:\n\\[ 35 - 2 = 33 \\]\nYes, that confirms it. Michael has 33 golf balls left at the end of Wednesday."
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Alright, let's tackle this problem about Olivia and her bagels. Okay, Olivia has $23 initially. She buys five bagels, and each bagel costs $3. I need to find out how much money she has left after her purchase. Hmm, seems straightforward, but let's make sure I get this right.\n\nFirst, I'll figure out the total cost of the bagels. If one bagel is $3, then five bagels would cost... wait, is it 5 times 3 or 3 times 5? Well, multiplication is commutative, so it doesn't matter, but just to be safe, I'll do it both ways.\n\nSo, \\(5 \\times 3 = 15\\) and \\(3 \\times 5 = 15\\). Okay, that checks out. The total cost is $15.\n\nNow, Olivia started with $23. After spending $15, how much does she have left? I'll subtract the amount spent from the initial amount.\n\n\\[23 - 15 = 8\\]\n\nSo, Olivia has $8 left. But wait, let me double-check that to be sure I didn't make a mistake.\n\nIf I add the amount spent and the amount left, it should equal the initial amount.\n\n\\[15 + 8 = 23\\]\n\nYes, that adds up correctly. Olivia spent $15 on bagels and has $8 remaining.\n\nBut just to explore another angle, what if I thought about it differently? Suppose Olivia buys the bagels one by one. After buying each bagel for $3, her remaining money decreases by $3 each time.\n\n1. After first bagel: \\(23 - 3 = 20\\)\n2. After second bagel: \\(20 - 3 = 17\\)\n3. After third bagel: \\(17 - 3 = 14\\)\n4. After fourth bagel: \\(14 - 3 = 11\\)\n5. After fifth bagel: \\(11 - 3 = 8\\)\n\nAgain, I end up with Olivia having $8 left. This reinforces that my initial calculation was correct.\n\nI wonder if there's a quicker way to see this. Maybe using division? If I divide the total money Olivia has by the cost per bagel, would that help?\n\n\\[23 \\div 3 \\approx 7.666\\]\n\nHmm, that doesn't seem directly useful here since she's buying a specific number of bagels rather than dividing her money evenly. It might be more relevant if she was trying to find out how many bagels she could buy with her money, but that's not the case in this problem.\n\nPerhaps using algebra? Let's set up an equation where \\(x\\) is the amount of money Olivia has left after buying the bagels.\n\n\\[23 - 5 \\times 3 = x\\]\n\nSimplifying:\n\n\\[23 - 15 = x\\]\n\n\\[x = 8\\]\n\nStill, I arrive at the same conclusion: Olivia has $8 left after her purchase.\n\nIs there a possibility that the problem expects the answer in a different form? Maybe in coins or something? The problem doesn't specify, so I'll stick with the straightforward answer.\n\nTo summarize, after buying five bagels at $3 each, Olivia spends a total of $15 and has:\n\n\\[\\boxed{8}\\]\n\ndollars left."
    }
]