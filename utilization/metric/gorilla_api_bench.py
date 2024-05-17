import numpy as np
import argparse
import json
from tree_sitter import Language, Parser
import tree_sitter_python as tspython


from .metric import Metric

from logging import getLogger
logger = getLogger(__name__)


class Gorilla_eval(Metric):
    def __init__(self, api_name):
        self.api_name = api_name

    def __call__(self, predictions, references):
        Accuracy, Hallucination = self.calculate_metrics(predictions, references)
        Accuracy = np.array(Accuracy)
        Hallucination = np.array(Hallucination)
        self._last_score_lists = {'Accuracy': Accuracy  , 'Hallucination rate': Hallucination}
        return {'Accuracy': np.mean(Accuracy), 'Hallucination rate': np.mean(Hallucination)}

    # Get all the subtrees given a root_node
    def get_all_sub_trees(self, root_node):
        node_stack = []
        sub_tree_sexp_list = []
        depth = 1
        text = root_node.text
        node_stack.append([root_node, depth])
        while len(node_stack) != 0:
            cur_node, cur_depth = node_stack.pop()
            if cur_node.child_count > 0:
                sub_tree_sexp_list.append([cur_node.sexp(), cur_depth, cur_node, cur_node.children[0].text])
            else:
                sub_tree_sexp_list.append([cur_node.sexp(), cur_depth, cur_node, None])
            for child_node in cur_node.children:
                if len(child_node.children) != 0:
                    depth = cur_depth + 1
                    node_stack.append([child_node, depth])
        return sub_tree_sexp_list

    # Parse the program into AST trees
    def ast_parse(self, candidate, lang='python'):
        LANGUAGE = Language(tspython.language(), "python")
        parser = Parser()
        parser.set_language(LANGUAGE)

        candidate_tree = parser.parse(bytes(candidate, 'utf8')).root_node
        return candidate_tree

    def get_args_hf(self, node):
        if node.child_count == 0:
            return []
        args_list = []
        for child in node.children[0].children[0].children[1].children:
            if "=" in child.text.decode():
                args_list.append(child.children[2].text)
            elif child.text.decode() != "(" and child.text.decode() != ")" and child.text.decode() != ",":
                args_list.append(child.text)
        return args_list

    def get_args_tf(self, node):
        if node.child_count == 0:
            return []
        args_list = []
        for child in node.children[0].children[0].children[1].children:
            if 'model=' in child.text.decode() or 'model =' in child.text.decode():
                args_list.append(child.children[2].text)
            elif child.text.decode() != "(" and child.text.decode() != ")" and child.text.decode() != ",":
                args_list.append(child.text)
        return args_list

    def get_args_th(self, node):
        if node.child_count == 0:
            return []
        args_list = []
        for child in node.children[0].children[0].children[1].children:
            if "repo_or_dir" in child.text.decode() or "model" in child.text.decode():
                args_list.append(child.children[2].text)
        return args_list

    # Check if there is an api match
    def ast_check(self, candidate_subtree_list, base_tree_list):
        for idx, base_tree in enumerate(base_tree_list):
            if base_tree.children[0].children[0].child_count == 0:
                continue
            api_name = base_tree.children[0].children[0].children[0].text
            for candidate_tree in candidate_subtree_list:
                if candidate_tree[3] == api_name:
                    break
            # Now we have a sub-tree
            candidate_tree = candidate_tree[2]

            if self.api_name == "torchhub":
                args_list = self.get_args_th(base_tree)
            elif self.api_name == "huggingface":
                args_list = self.get_args_hf(base_tree)
            elif self.api_name == "tensorflowhub":
                args_list = self.get_args_tf(base_tree)

            if len(args_list) == 0:
                continue
            ast_match = True
            for arg in args_list:
                if arg.decode().lstrip("'").rstrip("'") not in candidate_tree.text.decode():
                    ast_match = False
                    break
            if ast_match:
                return idx
        return -1

    # Parse the dataset
    def parse_dataset(self):
        # Read the api datasest
        api_database = []
        with open(f'./utilization/dataset/gorilla_data/data/api/{self.api_name}_api.jsonl', 'r') as f:
            for line in f:
                api_database.append(json.loads(line))

        # Parse all apis to ast trees
        ast_database = []
        for data in api_database:
            ast_tree = self.ast_parse(data['api_call'])
            ast_database.append(ast_tree)

        return api_database, ast_database


    def calculate_metrics(self, predictions, references):
        # Read datsets
        api_database, ast_database = self.parse_dataset()
        correct_list = []
        hallucination_list = []
        for idx, response in enumerate(predictions):
            try:
                output = response
            except:
                print('Error: cannot parse line ', idx)
                continue

            output = output.split("API_CALL")
            if len(output) == 1:
                api_call = output[0]
            else:
                # Parse the output
                output = output[1].split("API_PROVIDER")[0]
                if ":" not in output:
                    start = 0
                else:
                    start = output.index(":")
                if ")" not in output:
                    end = -2
                else:
                    end = output.rindex(")")
                api_call = output[start + 2:end + 1]

            # Parse the api_call into AST tree
            ast_tree = self.ast_parse(api_call)

            # Search for a subtree
            ast_subtree_list = self.get_all_sub_trees(ast_tree)

            # Check which ast tree is matching
            database_index = self.ast_check(ast_subtree_list, ast_database)
            # We cannot index this ast in our database
            if database_index == -1:
                hallucination_list.append(1)
                correct_list.append(0)
                continue
            hallucination_list.append(0)
            # We index our reference api_call
            ref_api_call = api_database[database_index] # 定位到对应的ref_api_call中
            # Check for functionality
            if ref_api_call['domain'] == eval(references[idx])['domain']:
                correct_list.append(1)
            else:
                correct_list.append(0)

        return correct_list, hallucination_list