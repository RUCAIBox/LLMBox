PAL_MATH_CHAT_PROMPT = [
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": '''```
def solution():
    """Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result
```

'''
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": '''```
def solution():
    """Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"""
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result
```

'''
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": '''```
def solution():
    """There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"""
    computers_initial = 9
    computers_per_day = 5
    num_days = 4  # 4 days between monday and thursday
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result
```

'''
    }
]

COT_EXAMPLARS = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6."
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5."
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39."
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8."
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9."
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29."
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33."
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8."
    }
]

LEAST_TO_MOST_EXAMPLARS = [
{
        "question": "Tom has 18 marbles. He gave 5 marbles to his friend Jerry. How many marbles does Tom have now?",
        "answer": """Let's analyze the situation: 1. How many marbles did Tom start with? 2. How many marbles does Tom have after giving 5 to Jerry?
1.Tom started with 18 marbles. 
2.After giving 5 marbles to Jerry, Tom has 18 - 5 = 13 marbles.
Final Answer: Tom has 13 marbles now."""
    },
    {
        "question": "Sarah had $50. She spent $20 on a book. How much money does Sarah have left?",
        "answer": """Breaking it down: 1. How much money did Sarah start with? 2. How much money does Sarah have after spending $20 on a book?
1.Sarah started with $50. 
2.After spending $20, Sarah has $50 - $20 = $30 left.
Final Answer: Sarah has $30 left."""
    },
    {
        "question": "Jason has a collection of 25 stamps. He bought 12 more stamps online. How many stamps does Jason have now?",
        "answer": """Let's break it into parts: 1. How many stamps did Jason start with? 2. How many stamps does Jason have after buying 12 more?
1.Jason started with 25 stamps. 
2.After buying 12 more, Jason has 25 + 12 = 37 stamps.
Final Answer: Jason has 37 stamps now."""
    },
    {
        "question": "Emily baked 40 cookies. She gave 15 cookies to her neighbor. How many cookies does Emily have left?",
        "answer": """Breaking it down: 1. How many cookies did Emily start with? 2. How many cookies does Emily have after giving 15 to her neighbor?
1.Emily started with 40 cookies. 
2.After giving 15 cookies away, Emily has 40 - 15 = 25 cookies left.
Final Answer: Emily has 25 cookies left."""
    },
    {
        "question": "Alex has 8 colorful balloons. He bought 3 more balloons at the store. How many balloons does Alex have in total?",
        "answer": """Let's analyze: 1. How many balloons did Alex start with? 2. How many balloons does Alex have after buying 3 more?
1.Alex started with 8 balloons. 
2.After buying 3 more, Alex has 8 + 3 = 11 balloons.
Final Answer: Alex has 11 balloons in total."""
    }
]