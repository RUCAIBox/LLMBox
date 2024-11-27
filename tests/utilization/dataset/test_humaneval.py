from utilization.dataset.humaneval import _truncate_code_at_stopwords, stop_words


def test_truncate_code_at_stopwords():
    code = {
        " \n    # Use a stack to keep track of the parentheses\n    stack = []\n    # Initialize the result list\n    result = []\n    # Iterate over the input string\n    for char in paren_string:\n        # If the character is an open parenthesis, push it onto the stack\n        if char == '(':\n            stack.append(char)\n        # If the character is a close parenthesis, pop the top of the stack\n        elif char == ')':\n            # If the stack is empty, the parenthesis is not balanced\n            if not stack:\n                result.append('')\n            else:\n                # Otherwise, pop the top of the stack\n                stack.pop()\n                result.append(char)\n    # Return the result list\n    return result": None,
        " \n    # Use a stack to keep track of the parentheses\n    stack = []\n    # Initialize the result list\n    result = []\n    # Iterate over the input string\n    for char in paren_string:\n        # If the character is an open parenthesis, push it onto the stack\n        if char == '(':\n            stack.append(char)\n        # If the character is a close parenthesis, pop the top of the stack\n        elif char == ')':\n            # If the stack is empty, the parenthesis is not balanced\n            if not stack:\n                result.append('')\n            else:\n                # Otherwise, pop the top of the stack\n                stack.pop()\n                result.append(char)\n    # Return the result list": "# No return statement found\n",
    }
    for code, expected in code.items():
        if expected is None:
            expected = code
        assert _truncate_code_at_stopwords(code, stop_words) == expected

