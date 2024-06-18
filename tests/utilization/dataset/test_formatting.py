import pytest

from ..fixtures import get_dataset, get_openai_model

keys = ["dataset_name", "num_shots", "cot", "ranking_type"]
datasets = {
    ("gsm8k", 0, None, None): (
        'Answer the following question.\n'
        '\n'
        'Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and '
        "bakes muffins for her friends every day with four. She sells the remainder at the farmers' "
        'market daily for $2 per fresh duck egg. How much in dollars does she make every day at the '
        "farmers' market?\n"
        'Answer:'
    ),
    ("gsm8k", 8, None, None): (
        'Answer the following question.\n'
        '\n'
        'Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. '
        'After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n'
        'Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So '
        'there must have been 21 - 15 = 6. So the answer is 6.\n'
        '\n'
        'Answer the following question.\n'
        '\n'
        'Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in '
        'the parking lot?\n'
        'Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.\n'
        '\n'
        'Answer the following question.\n'
        '\n'
        'Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they '
        'have left in total?\n'
        'Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = '
        '74. After eating 35, they had 74 - 35 = 39. So the answer is 39.\n'
        '\n'
        'Answer the following question.\n'
        '\n'
        'Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How '
        'many lollipops did Jason give to Denny?\n'
        'Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave '
        'Denny 20 - 12 = 8. So the answer is 8.\n'
        '\n'
        'Answer the following question.\n'
        '\n'
        'Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How '
        'many toys does he have now?\n'
        'Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 '
        'more toys. 5 + 4 = 9. So the answer is 9.\n'
        '\n'
        'Answer the following question.\n'
        '\n'
        'Question: There were nine computers in the server room. Five more computers were installed each '
        'day, from monday to thursday. How many computers are now in the server room?\n'
        'Answer: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 '
        '* 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.\n'
        '\n'
        'Answer the following question.\n'
        '\n'
        'Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 '
        'more. How many golf balls did he have at the end of wednesday?\n'
        'Answer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. '
        'After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.\n'
        '\n'
        'Answer the following question.\n'
        '\n'
        'Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have '
        'left?\n'
        'Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she '
        'has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.\n'
        '\n'
        'Answer the following question.\n'
        '\n'
        'Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and '
        "bakes muffins for her friends every day with four. She sells the remainder at the farmers' "
        'market daily for $2 per fresh duck egg. How much in dollars does she make every day at the '
        "farmers' market?\n"
        'Answer:'
    ),
    ("gsm8k", 8, "base", None): ("gsm8k", 8, None, None),  # alias: base cot -> None
    ("gsm8k", 2, "least_to_most", None): (
        'Answer the following question.\n'
        '\n'
        'Question: Elsa has 5 apples. Anna has 2 more apples than Elsa. How many apples do they have '
        'together?\n'
        'Answer: To answer the question "How many apples do they have together?", we need to know: "How '
        'many apples does Anna have?".\n'
        '1. Anna has 2 more apples than Elsa. So Anna has 2 + 5 = 7 apples.\n'
        '2. Elsa and Anna have 5 + 7 = 12 apples together. So the answer is 12.\n'
        '\n'
        'Answer the following question.\n'
        '\n'
        'Question: If Pam is currently twice as young as Rena is, and in 10 years Rena will be 5 years '
        'older than her, how old is Pam now?\n'
        'Answer: To answer the question "How old is Pam now?", we need to know: "How much older is Rena '
        'than Pam currently?".\n'
        '1. Since Rena will be 5 years older than Pam in 10 years, she must be 5 years older than Pam now '
        'as well.\n'
        '2. If Pam is currently twice as young as Rena, that means that Rena is currently twice as old as '
        "Pam is. So if P stands for Pam's age now and R stands for Rena's age now, then we know that R = "
        '2 * P And since Rena is 5 years older than Pam now, we know that R = P + 5. By substitution, we '
        'have P + 5 = 2 * P, which means that P = 5. So the answer is 5.\n'
        '\n'
        'Answer the following question.\n'
        '\n'
        'Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and '
        "bakes muffins for her friends every day with four. She sells the remainder at the farmers' "
        'market daily for $2 per fresh duck egg. How much in dollars does she make every day at the '
        "farmers' market?\n"
        'Answer:'
    ),
    ("gsm8k", 8, "least_to_most", None):
    ("gsm8k", 2, "least_to_most", None),  # alias: 8-shots -> 2-shots (len(example_data) == 2)
    ("gsm8k", 8, "pal", None): (
        "Let's use python to solve math problems. Here are some examples how to do it.\n"
        '\n'
        'Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have '
        'left?\n'
        'Answer:\n'
        '```\n'
        'def solution():\n'
        '    """Olivia has $23. She bought five bagels for $3 each. How much money does she have '
        'left?"""\n'
        '    money_initial = 23\n'
        '    bagels = 5\n'
        '    bagel_cost = 3\n'
        '    money_spent = bagels * bagel_cost\n'
        '    money_left = money_initial - money_spent\n'
        '    result = money_left\n'
        '    return result\n'
        '```\n'
        '\n'
        "Let's use python to solve math problems. Here are some examples how to do it.\n"
        '\n'
        'Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 '
        'more. How many golf balls did he have at the end of wednesday?\n'
        'Answer:\n'
        '```\n'
        'def solution():\n'
        '    """Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 '
        'more. How many golf balls did he have at the end of wednesday?"""\n'
        '    golf_balls_initial = 58\n'
        '    golf_balls_lost_tuesday = 23\n'
        '    golf_balls_lost_wednesday = 2\n'
        '    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n'
        '    result = golf_balls_left\n'
        '    return result\n'
        '```\n'
        '\n'
        "Let's use python to solve math problems. Here are some examples how to do it.\n"
        '\n'
        'Question: There were nine computers in the server room. Five more computers were installed each '
        'day, from monday to thursday. How many computers are now in the server room?\n'
        'Answer:\n'
        '```\n'
        'def solution():\n'
        '    """There were nine computers in the server room. Five more computers were installed each '
        'day, from monday to thursday. How many computers are now in the server room?"""\n'
        '    computers_initial = 9\n'
        '    computers_per_day = 5\n'
        '    num_days = 4  # 4 days between monday and thursday\n'
        '    computers_added = computers_per_day * num_days\n'
        '    computers_total = computers_initial + computers_added\n'
        '    result = computers_total\n'
        '    return result\n'
        '```\n'
        '\n'
        "Let's use python to solve math problems. Here are some examples how to do it.\n"
        '\n'
        'Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How '
        'many toys does he have now?\n'
        'Answer:\n'
        '```\n'
        'def solution():\n'
        '    """Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many '
        'toys does he have now?"""\n'
        '    toys_initial = 5\n'
        '    mom_toys = 2\n'
        '    dad_toys = 2\n'
        '    total_received = mom_toys + dad_toys\n'
        '    total_toys = toys_initial + total_received\n'
        '    result = total_toys\n'
        '    return result\n'
        '```\n'
        '\n'
        "Let's use python to solve math problems. Here are some examples how to do it.\n"
        '\n'
        'Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How '
        'many lollipops did Jason give to Denny?\n'
        'Answer:\n'
        '```\n'
        'def solution():\n'
        '    """Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How '
        'many lollipops did Jason give to Denny?"""\n'
        '    jason_lollipops_initial = 20\n'
        '    jason_lollipops_after = 12\n'
        '    denny_lollipops = jason_lollipops_initial - jason_lollipops_after\n'
        '    result = denny_lollipops\n'
        '    return result\n'
        '```\n'
        '\n'
        "Let's use python to solve math problems. Here are some examples how to do it.\n"
        '\n'
        'Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they '
        'have left in total?\n'
        'Answer:\n'
        '```\n'
        'def solution():\n'
        '    """Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they '
        'have left in total?"""\n'
        '    leah_chocolates = 32\n'
        '    sister_chocolates = 42\n'
        '    total_chocolates = leah_chocolates + sister_chocolates\n'
        '    chocolates_eaten = 35\n'
        '    chocolates_left = total_chocolates - chocolates_eaten\n'
        '    result = chocolates_left\n'
        '    return result\n'
        '```\n'
        '\n'
        "Let's use python to solve math problems. Here are some examples how to do it.\n"
        '\n'
        'Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and '
        "bakes muffins for her friends every day with four. She sells the remainder at the farmers' "
        'market daily for $2 per fresh duck egg. How much in dollars does she make every day at the '
        "farmers' market?\n"
        'Answer:'
    ),  # the first six examples get formatted because of max_example_tokens
    ("mmlu:abstract_algebra", 0, None, "ppl"): (
        'The following are multiple choice questions (with answers) about abstract algebra.\n'
        '\n'
        'Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\n'
        'A. 0\n'
        'B. 4\n'
        'C. 2\n'
        'D. 6\n'
        'Answer: A. 0'
    ),
    ("mmlu:abstract_algebra", 0, None, "ppl_no_option"): (
        'The following are multiple choice questions (with answers) about abstract algebra.\n'
        '\n'
        'Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\n'
        'Answer: 0'
    ),
    ("mmlu:abstract_algebra", 0, None, "prob"): (
        'The following are multiple choice questions (with answers) about abstract algebra.\n'
        '\n'
        'Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\n'
        'A. 0\n'
        'B. 4\n'
        'C. 2\n'
        'D. 6\n'
        'Answer:'
    ),
    ("mmlu:abstract_algebra", 0, None, "generation"): ("mmlu:abstract_algebra", 0, None, "prob"),
    ("mmlu:abstract_algebra", 5, None, "ppl"): (
        'The following are multiple choice questions (with answers) about abstract algebra.\n'
        '\n'
        'Question: Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\n'
        'A. 0\n'
        'B. 1\n'
        'C. 2\n'
        'D. 3\n'
        'Answer: B. 1\n'
        '\n'
        'The following are multiple choice questions (with answers) about abstract algebra.\n'
        '\n'
        'Question: Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement '
        '2 | If H and K are subgroups of G then HK is a subgroup of G.\n'
        'A. True, True\n'
        'B. False, False\n'
        'C. True, False\n'
        'D. False, True\n'
        'Answer: B. False, False\n'
        '\n'
        'The following are multiple choice questions (with answers) about abstract algebra.\n'
        '\n'
        'Question: Statement 1 | Every element of a group generates a cyclic subgroup of the group. '
        'Statement 2 | The symmetric group S_10 has 10 elements.\n'
        'A. True, True\n'
        'B. False, False\n'
        'C. True, False\n'
        'D. False, True\n'
        'Answer: C. True, False\n'
        '\n'
        'The following are multiple choice questions (with answers) about abstract algebra.\n'
        '\n'
        'Question: Statement 1| Every function from a finite set onto itself must be one to one. '
        'Statement 2 | Every subgroup of an abelian group is abelian.\n'
        'A. True, True\n'
        'B. False, False\n'
        'C. True, False\n'
        'D. False, True\n'
        'Answer: A. True, True\n'
        '\n'
        'The following are multiple choice questions (with answers) about abstract algebra.\n'
        '\n'
        'Question: Find the characteristic of the ring 2Z.\n'
        'A. 0\n'
        'B. 3\n'
        'C. 12\n'
        'D. 30\n'
        'Answer: A. 0\n'
        '\n'
        'The following are multiple choice questions (with answers) about abstract algebra.\n'
        '\n'
        'Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\n'
        'A. 0\n'
        'B. 4\n'
        'C. 2\n'
        'D. 6\n'
        'Answer: A. 0'
    ),
    ("mmlu:abstract_algebra", 5, None, "prob"): (
        'The following are multiple choice questions (with answers) about abstract algebra.\n'
        '\n'
        'Question: Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\n'
        'A. 0\n'
        'B. 1\n'
        'C. 2\n'
        'D. 3\n'
        'Answer: B\n'
        '\n'
        'The following are multiple choice questions (with answers) about abstract algebra.\n'
        '\n'
        'Question: Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement '
        '2 | If H and K are subgroups of G then HK is a subgroup of G.\n'
        'A. True, True\n'
        'B. False, False\n'
        'C. True, False\n'
        'D. False, True\n'
        'Answer: B\n'
        '\n'
        'The following are multiple choice questions (with answers) about abstract algebra.\n'
        '\n'
        'Question: Statement 1 | Every element of a group generates a cyclic subgroup of the group. '
        'Statement 2 | The symmetric group S_10 has 10 elements.\n'
        'A. True, True\n'
        'B. False, False\n'
        'C. True, False\n'
        'D. False, True\n'
        'Answer: C\n'
        '\n'
        'The following are multiple choice questions (with answers) about abstract algebra.\n'
        '\n'
        'Question: Statement 1| Every function from a finite set onto itself must be one to one. '
        'Statement 2 | Every subgroup of an abelian group is abelian.\n'
        'A. True, True\n'
        'B. False, False\n'
        'C. True, False\n'
        'D. False, True\n'
        'Answer: A\n'
        '\n'
        'The following are multiple choice questions (with answers) about abstract algebra.\n'
        '\n'
        'Question: Find the characteristic of the ring 2Z.\n'
        'A. 0\n'
        'B. 3\n'
        'C. 12\n'
        'D. 30\n'
        'Answer: A\n'
        '\n'
        'The following are multiple choice questions (with answers) about abstract algebra.\n'
        '\n'
        'Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\n'
        'A. 0\n'
        'B. 4\n'
        'C. 2\n'
        'D. 6\n'
        'Answer:'
    ),
}


@pytest.mark.parametrize("dataset_name, num_shots, cot, ranking_type", datasets.keys())
def test_formatting(get_dataset, dataset_name, num_shots, cot, ranking_type):

    dataset = get_dataset(dataset_name=dataset_name, num_shots=num_shots, cot=cot, ranking_type=ranking_type)
    eval_instance = datasets[(dataset_name, num_shots, cot, ranking_type)]
    while isinstance(eval_instance, tuple):
        eval_instance = datasets[eval_instance]

    formatted_instance = dataset.evaluation_instances[0]  # test the first instance
    if hasattr(formatted_instance, "to_model_prompt"):  # compat with Conversation
        formatted_instance = formatted_instance.to_model_prompt()
    formatted_instance = "".join(p for p in formatted_instance if isinstance(p, str))

    assert formatted_instance == eval_instance


def test_multi_turn(get_dataset):
    dataset = get_dataset(dataset_name="mt_bench", num_shots=0, ranking_type="generation")

    assert dataset.evaluation_instances[0].messages == [{
        "content":
        "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
        "role": "user",
    }]
    dataset.evaluation_instances[0].add_multi_turn(assistant="The fake model response.")
    assert dataset.evaluation_instances[0].messages == [{
        "content":
        "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
        "role": "user",
    }, {
        "content": "The fake model response.",
        "role": "assistant",
    }, {
        "content": "Rewrite your previous response. Start every sentence with the letter A.",
        "role": "user",
    }]

    dataset.evaluation_instances[0].add_multi_turn(assistant="A fake model response.")
    assert dataset.evaluation_instances[0].messages == [{
        "content":
        "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
        "role": "user",
    }, {
        "content": "The fake model response.",
        "role": "assistant",
    }, {
        "content": "Rewrite your previous response. Start every sentence with the letter A.",
        "role": "user",
    }, {
        "content": "A fake model response.",
        "role": "assistant",
    }]
