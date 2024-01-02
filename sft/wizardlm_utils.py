import openai
import os
import string

from llm_box.model.openai import Openai
from llm_box.utils import ModelArguments

openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_base = os.environ.get("OPENAI_API_BASE")

base_instruction_breath = "I want you act as a Prompt Creator.\r\n\
Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\r\n\
This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\r\n\
The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\r\n\
The #Created Prompt# must be reasonable and must be understood and responded by humans.\r\n\
'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\r\n"

base_instruction_depth = "I want you act as a Prompt Rewriter.\r\n \
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\r\n \
But the rewritten prompt must be reasonable and must be understood and responded by humans.\r\n \
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#. \r\n \
You SHOULD complicate the given prompt using the following method: \r\n\
{} \r\n\
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#. \r\n\
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\r\n"

gpt_compare_instruction_head = "Here are two Instructions to ChatGPT AI, do you think they are equal to each other, which meet the following requirements:\r\n\
1. They have same constraints and requirments.\r\n\
2. They have same depth and breadth of the inquiry.\r\n"

gpt_compare_instruction_tail = "Your Judgement (Just answer: Equal or Not Equal. No need to explain the reason.):\r\n"

common_stopwords = set(["the", "and", "is", "of", "in", "it", "you", "that", "for", "on", "with", "this"])


def createConstraintsPrompt(instruction):
    prompt = base_instruction_depth.format("Please add one more constraints/requirements into #The Given Prompt#'")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createDeepenPrompt(instruction):
    prompt = base_instruction_depth.format(
        "If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased."
    )
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createConcretizingPrompt(instruction):
    prompt = base_instruction_depth.format("Please replace general concepts with more specific concepts.")
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createReasoningPrompt(instruction):
    prompt = base_instruction_depth.format(
        "If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning."
    )
    prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Rewritten Prompt#:\r\n"
    return prompt


def createBreadthPrompt(instruction):
    prompt = base_instruction_breath
    prompt += "#Given Prompt#: \r\n {} \r\n".format(instruction)
    prompt += "#Created Prompt#:\r\n"
    return prompt


def call_chatgpt(instruction):
    model_args = ModelArguments(model_name_or_path='gpt-3.5-turbo-instruct')
    openai_instance = Openai(model_args)
    prompt = instruction,
    model_args_for_request = {
        'temperature': 1,
        'max_tokens': 2048,
        'top_p': 0.95,
        'frequency_penalty': 0,
        'presence_penalty': 0,
        'stop': None
    }

    response = openai_instance.request(prompt, model_args_for_request)
    res = ''
    res = response[0]['text']
    return res


def evol_elimination(seed_insinstruction, evol_instruction, response):

    # 1. use ChatGPT to compare the evolved instruction with seed.
    gpt_compare_instruction = gpt_compare_instruction_head \
        + "The First Prompt: <" + seed_insinstruction + ">\r\n" \
        + "The Second Prompt: <" + evol_instruction + ">\r\n" \
        + gpt_compare_instruction_tail
    judgement = call_chatgpt(gpt_compare_instruction)
    if 'not' not in judgement.lower():  # Equal instructions
        return False

    # 2. The evolved instruction makes it difficult for the LLM to generate a response.
    sorry_condition = "sorry" in response.lower()
    length_condition = len(response.split()) < 80
    if sorry_condition and length_condition:
        return False

    # 3. The response contains punctuation and stop words
    # common stop words(en)
    common_stopwords = set(["the", "and", "is", "of", "in", "it", "you", "that", "for", "on", "with", "this"])
    # Remove punctuation from the response
    response_no_punctuation = response.translate(str.maketrans('', '', string.punctuation))
    # Split the cleaned response into a list of words
    words = response_no_punctuation.split()
    only_punctuation_and_stopwords = all(
        word.lower() in common_stopwords or word in string.punctuation for word in words
    )
    if only_punctuation_and_stopwords:
        return False

    # 4. The evolved instruction copies some words from the evolving prompt
    # Phrases indicating copying from seed instruction
    copy_phrases = ["given prompt", "rewritten prompt", "#Rewritten Prompt#"]
    # Check if the evolved instruction contains any of the copy phrases
    copied_from_prompt = any(phrase.lower() in evol_instruction.lower() for phrase in copy_phrases)
    if copied_from_prompt:
        return False

    # 5. evol success!
    return True
