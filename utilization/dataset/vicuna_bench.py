from functools import cached_property

from ..metric import GPTEval
from .generation_dataset import GenerationDataset


class Vicuna_bench(GenerationDataset):
    """The dataset of Vicuna_bench.

    Following MT-bench, a multi-turn question set, using strong LLMs as judges to evaluate models on more open-ended questions.

    Example:
        question_id: 1
        category: "generic"
        turns: ["How can I improve my time management skills?"]
    """

    instruction = "{{turns[0]}}"
    example_set = None
    evaluation_set = ""
    load_args = ()
    metrics = [GPTEval()]

    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        for i in range(len(VICUNA_BENCH)):
            ref_answer = [
                d['choices'][0]['turns'][0] for d in REFERENCE_ANSWER
                if d['question_id'] == VICUNA_BENCH[i]["question_id"]
            ]
            if len(ref_answer) == 0:
                continue
            VICUNA_BENCH[i]['ref_answer_1'] = ref_answer[0]

        self.example_data = []
        self.evaluation_data = VICUNA_BENCH

    def format_instance(self, instance):
        return instance

    @cached_property
    def references(self):
        return self.evaluation_data


VICUNA_BENCH = [
    {
        "question_id": 1,
        "category": "generic",
        "turns": ["How can I improve my time management skills?"]
    },
    {
        "question_id": 2,
        "category": "generic",
        "turns": ["What are the most effective ways to deal with stress?"]
    },
    {
        "question_id": 3,
        "category": "generic",
        "turns": ["What are the main differences between Python and JavaScript programming languages?"]
    },
    {
        "question_id": 4,
        "category": "generic",
        "turns": ["How can I increase my productivity while working from home?"]
    },
    {
        "question_id": 5,
        "category": "generic",
        "turns": ["Can you explain the basics of quantum computing?"]
    },
    {
        "question_id": 6,
        "category": "generic",
        "turns": ["What are the differences between plant-based and animal-based protein sources?"]
    },
    {
        "question_id": 7,
        "category": "generic",
        "turns": ["How can I develop my critical thinking skills?"]
    },
    {
        "question_id": 8,
        "category": "generic",
        "turns": ["What are the major challenges faced by the education sector today?"]
    },
    {
        "question_id": 9,
        "category": "generic",
        "turns": ["What are the primary factors that influence consumer behavior?"]
    },
    {
        "question_id": 10,
        "category": "generic",
        "turns": ["What are the most effective strategies for conflict resolution in the workplace?"]
    },
    {
        "question_id":
        11,
        "category":
        "knowledge",
        "turns": [
            "What are some potential implications of using a single-use plastic bottle versus a reusable bottle on both the environment and human health?"
        ]
    },
    {
        "question_id":
        12,
        "category":
        "knowledge",
        "turns":
        ["What factors would you consider when designing an inclusive and accessible public transportation system?"]
    },
    {
        "question_id": 13,
        "category": "knowledge",
        "turns": ["How can governments utilize fiscal and monetary policies to combat economic recessions?"]
    },
    {
        "question_id":
        14,
        "category":
        "knowledge",
        "turns": [
            "How do language and cultural barriers affect the way people communicate and form relationships in multicultural societies?"
        ]
    },
    {
        "question_id":
        15,
        "category":
        "knowledge",
        "turns": [
            "Describe a scenario where artificial intelligence could be used to improve the quality and efficiency of healthcare delivery."
        ]
    },
    {
        "question_id":
        16,
        "category":
        "knowledge",
        "turns": [
            "Explain the process of gene editing using CRISPR-Cas9 technology, and discuss its potential applications and ethical implications."
        ]
    },
    {
        "question_id":
        17,
        "category":
        "knowledge",
        "turns": [
            "How do vaccinations work to protect individuals and communities from infectious diseases, and what is herd immunity?"
        ]
    },
    {
        "question_id":
        18,
        "category":
        "knowledge",
        "turns": [
            "How do social media platforms influence the way people consume and share news, and what are the potential implications for the spread of misinformation?"
        ]
    },
    {
        "question_id":
        19,
        "category":
        "knowledge",
        "turns": [
            "How do cultural, social, and economic factors influence people's food choices, and how can this knowledge be used to promote healthier diets?"
        ]
    },
    {
        "question_id":
        20,
        "category":
        "knowledge",
        "turns":
        ["Explain the process of natural selection and how it contributes to the evolution and adaptation of species."]
    },
    {
        "question_id": 21,
        "category": "roleplay",
        "turns": ["How would you introduce yourself as a medieval knight at a royal banquet?"]
    },
    {
        "question_id": 22,
        "category": "roleplay",
        "turns":
        ["As a pirate captain, what would you say to your crew to motivate them to search for hidden treasure?"]
    },
    {
        "question_id": 23,
        "category": "roleplay",
        "turns": ["If you were a Shakespearean character, how would you declare your love for someone in a soliloquy?"]
    },
    {
        "question_id": 24,
        "category": "roleplay",
        "turns": ["As a superhero, how would you explain your origin story to a curious child?"]
    },
    {
        "question_id":
        25,
        "category":
        "roleplay",
        "turns": [
            "Imagine you are a time traveler from the year 3000. What technological advancements would you tell people about?"
        ]
    },
    {
        "question_id": 26,
        "category": "roleplay",
        "turns": ["As a sports commentator, describe the winning play in the final seconds of a championship game."]
    },
    {
        "question_id": 27,
        "category": "roleplay",
        "turns":
        ["Pretend to be a world-famous chef. How would you describe your signature dish to a panel of judges?"]
    },
    {
        "question_id":
        28,
        "category":
        "roleplay",
        "turns": [
            "You are a mountain climber reaching the summit of Mount Everest. Describe your emotions and the view from the top."
        ]
    },
    {
        "question_id":
        29,
        "category":
        "roleplay",
        "turns":
        ["As a space colonist on Mars, describe your daily life and the challenges you face living on another planet."]
    },
    {
        "question_id":
        30,
        "category":
        "roleplay",
        "turns": [
            "Pretend to be a character in a post-apocalyptic world. Describe how you survive and the allies you encounter."
        ]
    },
    {
        "question_id":
        31,
        "category":
        "common-sense",
        "turns": [
            "How can you determine if a restaurant is popular among locals or mainly attracts tourists, and why might this information be useful?"
        ]
    },
    {
        "question_id":
        32,
        "category":
        "common-sense",
        "turns": [
            "What are some subtle clues that suggest someone is pretending to understand a topic or conversation when they are actually confused or uninformed?"
        ]
    },
    {
        "question_id":
        33,
        "category":
        "common-sense",
        "turns": [
            "Why might someone choose to use a paper map or ask for directions instead of relying on a GPS device or smartphone app?"
        ]
    },
    {
        "question_id": 34,
        "category": "common-sense",
        "turns":
        ["How can you determine if a person is genuinely interested in a conversation or simply being polite?"]
    },
    {
        "question_id":
        35,
        "category":
        "common-sense",
        "turns": [
            "Why might someone prefer to shop at a small, locally-owned business instead of a large chain store, even if the prices are higher?"
        ]
    },
    {
        "question_id":
        36,
        "category":
        "common-sense",
        "turns": [
            "How can you assess the credibility of a source of information, such as a news article or blog post, without relying solely on the reputation of the author or publisher?"
        ]
    },
    {
        "question_id":
        37,
        "category":
        "common-sense",
        "turns": [
            "Why do some people enjoy the sensation of being scared, such as by watching horror movies or going on roller coasters, while others avoid these experiences?"
        ]
    },
    {
        "question_id":
        38,
        "category":
        "common-sense",
        "turns": [
            "How can observing the behavior of other people in a social situation provide clues about cultural norms and expectations?"
        ]
    },
    {
        "question_id": 39,
        "category": "common-sense",
        "turns":
        ["Do we have a moral obligation to explore space, or should we focus on solving Earth's problems first?"]
    },
    {
        "question_id":
        40,
        "category":
        "common-sense",
        "turns": [
            "In a world where automation is becoming increasingly prevalent, is it more important to prioritize job creation or technological progress?"
        ]
    },
    {
        "question_id":
        41,
        "category":
        "fermi",
        "turns": [
            "How many times does the average human blink in a lifetime? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."
        ]
    },
    {
        "question_id":
        42,
        "category":
        "fermi",
        "turns": [
            "How many atoms are in a grain of salt? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."
        ]
    },
    {
        "question_id":
        43,
        "category":
        "fermi",
        "turns": [
            "How many lightning strikes occur on Earth each day? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."
        ]
    },
    {
        "question_id":
        44,
        "category":
        "fermi",
        "turns": [
            "How many balloons would it take to lift a house like in the movie \"Up\"? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."
        ]
    },
    {
        "question_id":
        45,
        "category":
        "fermi",
        "turns": [
            "How many text messages are sent globally in a minute? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."
        ]
    },
    {
        "question_id":
        46,
        "category":
        "fermi",
        "turns": [
            "How many words are spoken daily on Earth? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."
        ]
    },
    {
        "question_id":
        47,
        "category":
        "fermi",
        "turns": [
            "How many snowflakes fall during a typical winter? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."
        ]
    },
    {
        "question_id":
        48,
        "category":
        "fermi",
        "turns": [
            "How many pages are in all the books ever written? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."
        ]
    },
    {
        "question_id":
        49,
        "category":
        "fermi",
        "turns": [
            "How many times has the Earth orbited the Sun since the beginning of life? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."
        ]
    },
    {
        "question_id":
        50,
        "category":
        "fermi",
        "turns": [
            "How many songs have been recorded throughout history? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step."
        ]
    },
    {
        "question_id": 51,
        "category": "counterfactual",
        "turns": ["What if the Internet had been invented during the Renaissance period?"]
    },
    {
        "question_id": 52,
        "category": "counterfactual",
        "turns": ["What if the Aztecs had successfully repelled the Spanish conquistadors?"]
    },
    {
        "question_id": 53,
        "category": "counterfactual",
        "turns": ["What if the Black Death had not occurred in the 14th century?"]
    },
    {
        "question_id": 54,
        "category": "counterfactual",
        "turns": ["What if Isaac Newton had focused on biology instead of physics?"]
    },
    {
        "question_id": 55,
        "category": "counterfactual",
        "turns": ["What if the Beatles had never formed as a band?"]
    },
    {
        "question_id": 56,
        "category": "counterfactual",
        "turns": ["What if Alan Turing had not cracked the Enigma code during World War II?"]
    },
    {
        "question_id": 57,
        "category": "counterfactual",
        "turns": ["What if the Suez Canal had never been constructed?"]
    },
    {
        "question_id": 58,
        "category": "counterfactual",
        "turns": ["What if the Maya civilization had never mysteriously collapsed?"]
    },
    {
        "question_id": 59,
        "category": "counterfactual",
        "turns": ["What if Christopher Columbus had not discovered the Americas?"]
    },
    {
        "question_id": 60,
        "category": "counterfactual",
        "turns": ["What if Vincent van Gogh had been a successful artist during his lifetime?"]
    },
    {
        "question_id":
        61,
        "category":
        "coding",
        "turns": [
            "Develop a C++ program that reads a text file line by line and counts the number of occurrences of a specific word in the file."
        ]
    },
    {
        "question_id":
        62,
        "category":
        "coding",
        "turns": [
            "Implement a Python function to find the longest common subsequence of two input strings using dynamic programming."
        ]
    },
    {
        "question_id": 63,
        "category": "coding",
        "turns": ["Implement a regular expression in Python to validate an email address."]
    },
    {
        "question_id": 64,
        "category": "coding",
        "turns": ["Write a program to find the nth Fibonacci number using dynamic programming."]
    },
    {
        "question_id": 65,
        "category": "coding",
        "turns": ["Implement a binary search algorithm to find a specific element in a sorted array."]
    },
    {
        "question_id": 66,
        "category": "coding",
        "turns": ["Implement a queue data structure using two stacks in Python."]
    },
    {
        "question_id": 67,
        "category": "coding",
        "turns":
        ["Implement a program to find the common elements in two arrays without using any extra data structures."]
    },
    {
        "question_id": 68,
        "category": "math",
        "turns": ["Given that f(x) = 5x^3 - 2x + 3, find the value of f(2)."]
    },
    {
        "question_id": 69,
        "category": "math",
        "turns": ["Solve for x in the equation 3x + 10 = 5(x - 2)."]
    },
    {
        "question_id": 70,
        "category": "math",
        "turns": ["If the endpoints of a line segment are (2, -2) and (10, 4), what is the length of the segment?"]
    },
    {
        "question_id": 71,
        "category": "writing",
        "turns": ["Can you help me write a formal email to a potential business partner proposing a joint venture?"]
    },
    {
        "question_id":
        72,
        "category":
        "writing",
        "turns": [
            "Can you help me write a resignation letter to my current employer, while leaving on good terms and expressing gratitude for the opportunities provided?"
        ]
    },
    {
        "question_id":
        73,
        "category":
        "writing",
        "turns": [
            "Use an appropriate format to structure a formal letter of recommendation for a student applying to a prestigious graduate program in computer science."
        ]
    },
    {
        "question_id":
        74,
        "category":
        "writing",
        "turns":
        ["Write a compelling product launch announcement email to inform our customers of our new software solution."]
    },
    {
        "question_id":
        75,
        "category":
        "writing",
        "turns": [
            "Draft an apology email to a customer who experienced a delay in their order, and provide reassurance that the issue has been resolved."
        ]
    },
    {
        "question_id": 76,
        "category": "writing",
        "turns": ["Write a script for a YouTube video exploring the history and cultural significance of jazz."]
    },
    {
        "question_id":
        77,
        "category":
        "writing",
        "turns": [
            "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
        ]
    },
    {
        "question_id":
        78,
        "category":
        "writing",
        "turns": [
            "Write a captivating movie review for a recently released science fiction film, discussing its plot, characters, and special effects."
        ]
    },
    {
        "question_id":
        79,
        "category":
        "writing",
        "turns": [
            "Structure a podcast script for an episode discussing the influence of streaming platforms on the music industry."
        ]
    },
    {
        "question_id":
        80,
        "category":
        "writing",
        "turns":
        ["Write a symphony concert review, discussing the orchestra's performance and overall audience experience."]
    },
]

REFERENCE_ANSWER = [{
    "question_id":
    61,
    "answer_id":
    "YdL4XwENkLCLXMbH65rjKy",
    "model_id":
    "gpt-4",
    "choices": [{
        "index":
        0,
        "turns": [
            "Here's a simple C++ program that reads a text file line by line and counts the number of occurrences of a specific word in the file:\n\n```cpp\n#include <iostream>\n#include <fstream>\n#include <string>\n#include <sstream>\n\nint countWordOccurrences(const std::string& line, const std::string& word) {\n    std::istringstream iss(line);\n    std::string currentWord;\n    int count = 0;\n\n    while (iss >> currentWord) {\n        if (currentWord == word) {\n            count++;\n        }\n    }\n\n    return count;\n}\n\nint main() {\n    std::string fileName, wordToCount;\n    std::cout << \"Enter the file name: \";\n    std::cin >> fileName;\n    std::cout << \"Enter the word to count: \";\n    std::cin >> wordToCount;\n\n    std::ifstream inputFile(fileName);\n    if (!inputFile) {\n        std::cerr << \"Error opening the file: \" << fileName << std::endl;\n        return 1;\n    }\n\n    std::string line;\n    int totalCount = 0;\n    while (std::getline(inputFile, line)) {\n        totalCount += countWordOccurrences(line, wordToCount);\n    }\n\n    inputFile.close();\n\n    std::cout << \"The word \\\"\" << wordToCount << \"\\\" occurs \" << totalCount << \" times in the file.\" << std::endl;\n\n    return 0;\n}\n```\n\nTo compile and run the program, save it to a file named `word_counter.cpp` and use the following commands:\n\n```sh\ng++ -o word_counter word_counter.cpp\n./word_counter\n```\n\nThe program will prompt you to enter the file name and the word you want to count. After that, it will display the number of occurrences of the specified word in the file."
        ]
    }],
    "tstamp":
    1686545085.5954556
}, {
    "question_id":
    62,
    "answer_id":
    "4nk9zQkeRKvfjxEEqfyXmj",
    "model_id":
    "gpt-4",
    "choices": [{
        "index":
        0,
        "turns": [
            "Here's a Python function that finds the longest common subsequence of two input strings using dynamic programming:\n\n```python\ndef longest_common_subsequence(s1, s2):\n    # Create a matrix to store the lengths of common subsequences\n    matrix = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]\n\n    # Iterate through the strings and fill the matrix\n    for i in range(len(s1) - 1, -1, -1):\n        for j in range(len(s2) - 1, -1, -1):\n            if s1[i] == s2[j]:\n                matrix[i][j] = 1 + matrix[i + 1][j + 1]\n            else:\n                matrix[i][j] = max(matrix[i + 1][j], matrix[i][j + 1])\n\n    # Reconstruct the longest common subsequence from the matrix\n    lcs = []\n    i, j = 0, 0\n    while i < len(s1) and j < len(s2):\n        if s1[i] == s2[j]:\n            lcs.append(s1[i])\n            i += 1\n            j += 1\n        elif matrix[i + 1][j] > matrix[i][j + 1]:\n            i += 1\n        else:\n            j += 1\n\n    return ''.join(lcs)\n\n# Example usage:\ns1 = \"ABCDGH\"\ns2 = \"AEDFHR\"\nprint(longest_common_subsequence(s1, s2))  # Output: \"ADH\"\n```\n\nThis function uses dynamic programming to build a matrix that stores the lengths of common subsequences for each pair of substrings. It then reconstructs the longest common subsequence from the matrix."
        ]
    }],
    "tstamp":
    1686544775.161658
}, {
    "question_id":
    63,
    "answer_id":
    "2MkXEfsoSuSC72BYGJuqhX",
    "model_id":
    "gpt-4",
    "choices": [{
        "index":
        0,
        "turns": [
            "To implement a regular expression in Python to validate an email address, you can use the `re` module. Here's a simple example:\n\n```python\nimport re\n\ndef validate_email(email):\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))\n\n# Test cases\nemails = [\n    \"test@example.com\",\n    \"test@example.co.uk\",\n    \"test+123@example.com\",\n    \"test@subdomain.example.com\",\n    \"test@sub_domain.example.com\",\n    \"test@sub-domain.example.com\",\n    \"test@example\",\n    \"test@.com\",\n    \"test@.example.com\",\n    \"test@example..com\",\n    \"test@.example..com\",\n    \"test@example.com.\",\n    \".test@example.com\",\n]\n\nfor email in emails:\n    print(f\"{email}: {validate_email(email)}\")\n```\n\nThis regular expression checks for the following conditions:\n\n1. The email address starts with one or more alphanumeric characters, dots, underscores, percent signs, plus signs, or hyphens.\n2. Followed by the '@' symbol.\n3. Followed by one or more alphanumeric characters, dots, or hyphens.\n4. Followed by a dot.\n5. Ends with two or more alphabetic characters.\n\nPlease note that this is a simple example and may not cover all edge cases. You can modify the regular expression to better suit your specific requirements."
        ]
    }],
    "tstamp":
    1686544778.904383
}, {
    "question_id":
    64,
    "answer_id":
    "jPeywxk9g4LWyLPurjhsd2",
    "model_id":
    "gpt-4",
    "choices": [{
        "index":
        0,
        "turns": [
            "Here's a Python program to find the nth Fibonacci number using dynamic programming (also known as memoization):\n\n```python\ndef fibonacci(n, memo={}):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    elif n not in memo:\n        memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)\n    return memo[n]\n\n# Test the function\nn = int(input(\"Enter the value of n: \"))\nresult = fibonacci(n)\nprint(f\"The {n}th Fibonacci number is: {result}\")\n```\n\nThis program defines a function `fibonacci(n, memo={})` that calculates the nth Fibonacci number using dynamic programming. The `memo` dictionary is used to store the results of previously calculated Fibonacci numbers, which helps to avoid redundant calculations and improve the efficiency of the program."
        ]
    }],
    "tstamp":
    1686544763.1005273
}, {
    "question_id":
    65,
    "answer_id":
    "hD8NiB3PY5aoWg95mNjnj3",
    "model_id":
    "gpt-4",
    "choices": [{
        "index":
        0,
        "turns": [
            "Here's a Python implementation of the binary search algorithm to find a specific element in a sorted array:\n\n```python\ndef binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n\n    while low <= high:\n        mid = (low + high) // 2\n        mid_val = arr[mid]\n\n        if mid_val == target:\n            return mid\n        elif mid_val < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n\n    return -1\n\n# Example usage:\narr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]\ntarget = 11\n\nresult = binary_search(arr, target)\nif result != -1:\n    print(f\"Element {target} found at index {result}\")\nelse:\n    print(f\"Element {target} not found in the array\")\n```\n\nThis binary search function takes a sorted array `arr` and a target value `target` as input. It initializes two pointers, `low` and `high`, to the first and last indices of the array, respectively. The function then iteratively updates the pointers and calculates the middle index `mid` until the target value is found or the search space is exhausted. If the target value is found, the function returns the index of the target value; otherwise, it returns -1."
        ]
    }],
    "tstamp":
    1686544779.33291
}, {
    "question_id":
    66,
    "answer_id":
    "c4rAXsTfmioefx6icubuFB",
    "model_id":
    "gpt-4",
    "choices": [{
        "index":
        0,
        "turns": [
            "To implement a queue using two stacks in Python, you can use the following code:\n\n```python\nclass Queue:\n    def __init__(self):\n        self.stack1 = []\n        self.stack2 = []\n\n    def enqueue(self, item):\n        self.stack1.append(item)\n\n    def dequeue(self):\n        if not self.stack2:\n            while self.stack1:\n                self.stack2.append(self.stack1.pop())\n        return self.stack2.pop()\n\n    def is_empty(self):\n        return not self.stack1 and not self.stack2\n\n    def size(self):\n        return len(self.stack1) + len(self.stack2)\n\n\n# Example usage:\nqueue = Queue()\nqueue.enqueue(1)\nqueue.enqueue(2)\nqueue.enqueue(3)\n\nprint(queue.dequeue())  # Output: 1\nprint(queue.dequeue())  # Output: 2\n\nqueue.enqueue(4)\nqueue.enqueue(5)\n\nprint(queue.dequeue())  # Output: 3\nprint(queue.dequeue())  # Output: 4\nprint(queue.dequeue())  # Output: 5\n```\n\nIn this implementation, we use two stacks (`stack1` and `stack2`) to simulate the behavior of a queue. When enqueuing an item, we simply push it onto `stack1`. When dequeuing an item, we pop from `stack2`. If `stack2` is empty, we first transfer all items from `stack1` to `stack2` (reversing their order) before popping from `stack2`. This ensures that the oldest item in the queue is always at the top of `stack2`."
        ]
    }],
    "tstamp":
    1686544798.8350058
}, {
    "question_id":
    67,
    "answer_id":
    "UQ3TFrzVqXtMTWpvhqtGym",
    "model_id":
    "gpt-4",
    "choices": [{
        "index":
        0,
        "turns": [
            "You can implement a program to find the common elements in two arrays without using any extra data structures by using nested loops. Here's an example in Python:\n\n```python\ndef find_common_elements(arr1, arr2):\n    common_elements = []\n\n    for i in arr1:\n        for j in arr2:\n            if i == j:\n                common_elements.append(i)\n                break\n\n    return common_elements\n\n# Example usage:\narr1 = [1, 2, 3, 4, 5]\narr2 = [4, 5, 6, 7, 8]\n\ncommon_elements = find_common_elements(arr1, arr2)\nprint(\"Common elements:\", common_elements)\n```\n\nThis program defines a function `find_common_elements` that takes two arrays as input. It initializes an empty list `common_elements` to store the common elements found. Then, it iterates through each element in the first array using a loop, and for each element, it iterates through the second array using another loop. If an element from the first array is found in the second array, it is added to the `common_elements` list and the inner loop is terminated using the `break` statement.\n\nFinally, the function returns the `common_elements` list. In the example usage, the program finds the common elements between two arrays `arr1` and `arr2`, and prints the result."
        ]
    }],
    "tstamp":
    1686544805.0822344
}, {
    "question_id":
    68,
    "answer_id":
    "eBGY6U9KCAsfuedcASSK7Q",
    "model_id":
    "gpt-4",
    "choices": [{
        "index":
        0,
        "turns": [
            "To find the value of f(2), substitute x = 2 into the function f(x) = 5x^3 - 2x + 3.\n\nf(2) = 5(2)^3 - 2(2) + 3\nf(2) = 5(8) - 4 + 3\nf(2) = 40 - 4 + 3\nf(2) = 36 + 3\nf(2) = 39\n\nSo, the value of f(2) is 39."
        ]
    }],
    "tstamp":
    1686544893.1474469
}, {
    "question_id":
    69,
    "answer_id":
    "KsgnWzbpEvoW7RmRwaLgMT",
    "model_id":
    "gpt-4",
    "choices": [{
        "index":
        0,
        "turns": [
            "To solve for x in the equation 3x + 10 = 5(x - 2), follow these steps:\n\n1. Distribute the 5 on the right side of the equation:\n3x + 10 = 5x - 10\n\n2. Move the x terms to one side by subtracting 3x from both sides:\n10 = 2x - 10\n\n3. Move the constant term to the other side by adding 10 to both sides:\n20 = 2x\n\n4. Divide both sides by 2 to isolate x:\nx = 10\n\nSo, the solution is x = 10."
        ]
    }],
    "tstamp":
    1686544795.032636
}, {
    "question_id":
    70,
    "answer_id":
    "BT82nWd8ZtXTzHrvhkg6eb",
    "model_id":
    "gpt-4",
    "choices": [{
        "index":
        0,
        "turns": [
            "To find the length of the line segment, we can use the distance formula:\n\nDistance = \u221a((x2 - x1)^2 + (y2 - y1)^2)\n\nHere, (x1, y1) = (2, -2) and (x2, y2) = (10, 4).\n\nPlugging in the values, we get:\n\nDistance = \u221a((10 - 2)^2 + (4 - (-2))^2)\nDistance = \u221a((8)^2 + (6)^2)\nDistance = \u221a(64 + 36)\nDistance = \u221a100\nDistance = 10\n\nSo, the length of the line segment is 10 units."
        ]
    }],
    "tstamp":
    1686544809.1491597
}]
