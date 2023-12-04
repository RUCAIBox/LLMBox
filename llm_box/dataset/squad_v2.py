from .generation_dataset import GenerationDataset
import re
from ..metric import F1, Em
import numpy as np


class Squad_v2(GenerationDataset):
    """The dataset of Squad_v2.

    Gcombines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. 

    Examples:
        context: The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.
        question: In what country is Normandy located?
        answer: ['France', 'France', 'France', 'France']
    """

    name = "squad_v2"
    instruction = "Answer the question based on the given passage."
    evaluation_type = "user_defined"
    example_set = "train"
    evaluation_set = "validation"

    load_args = ("squad_v2",)
    metrics = [F1(), Em()]

    def evaluation(self, batch):
        self.get_ppl = self.model.get_ppl
        self.generation = self.model.generation
        ppl_prompt = [option for _, ppl_batch in batch for option in ppl_batch]
        generation_prompt = [generation_batch for generation_batch, _ in batch]
        ppls = self.get_ppl(ppl_prompt)
        answers = self.generation(generation_prompt)
        ppls = [(ppls[i], ppls[i + 1]) for i in range(0, len(ppls), 2)]
        return list(zip(answers, ppls))

    def format_instance(self, instance):
        source_text = "Title: " + instance["title"]  + "\n\nBackground: " \
            + instance["context"] + "\n\nQ: " + instance["question"] + "\n\nA:"
        text = instance["answers"]["text"]
        if not text:
            text = "The question is not answerable."
        else:
            text = text[0]
        target_text = " " + text
        return dict(source=source_text, target=target_text)

    def construct_instances(self):
        r"""Construct and format all the instances of `evaluation_data`.

        Returns:
            List[str]: The list of final formatted instances.
        """
        self.evaluation_instances = []
        self.option_nums = []
        for instance in self.evaluation_data:
            formatted_instance = self.format_instance(instance)
            generation_formatted_instance = self.format_instruction_and_examples(formatted_instance['source'])
            ppl_formatted_instance = [
                self.format_instruction_and_examples(
                    formatted_instance["source"][:-2] + "Can this question be answered? Yes or no?", option
                ) for option in [" No.", " Yes."]
            ]
            self.evaluation_instances.append((generation_formatted_instance, ppl_formatted_instance))
        self.evaluation_instances = self.evaluation_instances * self.args.sample_num
        self.option_nums = self.option_nums * self.args.sample_num

    def construct_examples(self, instance=None) -> str:
        r"""Format one instance with the instruction and demonstration.

        Args:
            instance (Dict): a pre-formatted evaluation instance.

        Returns:
            str: The constructed demonstration text.
        """
        if self.num_shots == 0:
            return ""
        elif len(self.example_data) == 0:
            raise ValueError(
                f"Receive num_shots={self.num_shots}, but cannot construct examples for dataset {self.name} without example data."
            )

        # selection algorithm
        # TODO: ICL
        indice = np.random.choice(len(self.example_data), self.args.num_shots)

        # TODO: tokenizer efficiency
        # construct few-shot examples
        generation_example_text = ""
        generation_example_token_nums = 0
        ppl_example_text = ""
        ppl_example_token_nums = 0
        for index in indice:
            example = self.format_instance(self.example_data[index])
            cur_example_text = self.args.instance_format.format_map(example) + "\n\n"
            cur_token_num = len(self.tokenizer.encode(cur_example_text))
            if cur_token_num + generation_example_token_nums <= self.max_example_tokens:
                generation_example_text += cur_example_text
                generation_example_token_nums += cur_token_num
            example["source"] = example["source"][:-2] + "Can this question be answered? Yes or no?"
            example["target"] = " No." if example["target"] == ' The question is not answerable.' else ' Yes.'
            cur_example_text = self.args.instance_format.format_map(example) + "\n\n"
            cur_token_num = len(self.tokenizer.encode(cur_example_text))
            if cur_token_num + ppl_example_token_nums <= self.max_example_tokens:
                ppl_example_text += cur_example_text
                ppl_example_token_nums += cur_token_num

        return generation_example_text, ppl_example_text

    def format_instruction_and_examples(self, source, target=""):
        r"""Format one instance with the instruction and demonstration.

        Args:
            source (str): the pre-formatted source text.
            target (str, optional): the pre-formatted target text (default to "").

        Returns:
            Union[str, Tuple(str, str)]: The final formatted instance.
        """
        # TODO: instruction template
        # TODO: ICL
        examples = self.examples
        if self.num_shots != 0:
            examples = examples[1] if target else examples[0]

        if self.model.type == 'base':
            source = examples + source
        elif self.model.type == 'instruction':
            source = self.instruction + "\n\n" + examples + source

        if target:
            return source, target
        else:
            return source

    @property
    def references(self):
        return [
            instance["answers"]["text"] if instance["answers"]["text"] else ['unanswerable']
            for instance in self.evaluation_data
        ]

    @staticmethod
    def post_processing(preds):
        predictions = []
        pattern = r'[.!(\n)]'
        for pred in preds:
            generation_pred, ppl_pred = pred
            match = re.search(pattern, generation_pred)
            if match:
                index = match.start()
                generation_pred = generation_pred[:index]
            ppl_pred = np.array([result / length for result, length in ppl_pred]).argmin()
            predictions.append(generation_pred if ppl_pred == 1 else "unanswerable")
        print(predictions)
        return predictions
