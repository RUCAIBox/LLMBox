from .generation_dataset import GenerationDataset
import re
from ..metric import F1, Em
import numpy as np


class Squad_v2(GenerationDataset):
    """The dataset of GSM8K.

    GSM8K(Cobbe et al. 2021), linguistically diverse grade school math word problems

    Examples:
        question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
        answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72
    """


    name = "squad_v2"
    instruction = "Answer the question based on the given passage."
    
    example_set = "train"
    evaluation_set = "validation"
    
    load_args = ("squad_v2",)
    metrics = [F1(symbol_for_not_answer=[]),Em(symbol_for_not_answer=[])]

    def __init__(self, args, model, subset_name):
        super().__init__(args, model, subset_name)
        self.get_ppl = self.model.get_ppl
        self.generation = self.model.generation
        self.model.generation = self._generation

    def _generation(self, batch):
        ppl_prompt = [option for _ ,ppl_batch in batch for option in ppl_batch]
        generation_prompt = [generation_batch for generation_batch, _ in batch]
        ppls = self.get_ppl(ppl_prompt)
        answers =  self.generation(generation_prompt)
        ppls = [(ppls[i],ppls[i+1]) for i in range(0,len(ppls),2)]
        return list(zip(answers,ppls))
    

    def format_instance(self, instance):
        source_text = "Title: " + instance["title"]  + "\n\nBackground: " \
            + instance["context"] + "\n\nQ: " + instance["question"] + "\n\nA:"
        text = instance["answers"]["text"]
        if not text:
            text = "The question is not answerable."
        else:
            text = text[0]
        target_text = " " + text
        return dict(
        source = source_text,
        target = target_text
        )
    
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
            ppl_formatted_instance = [self.format_instruction_and_examples(formatted_instance["source"][:-2] +"Can this question be answered? Yes or no?", option) for option in [" No."," Yes."]]
            self.evaluation_instances.append((generation_formatted_instance, ppl_formatted_instance))
        self.evaluation_instances = self.evaluation_instances * self.args.sample_num
        self.option_nums = self.option_nums * self.args.sample_num

    @property
    def references(self):
        return [instance["answers"]["text"] for instance in self.evaluation_data]

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
            predictions.append((generation_pred,ppl_pred))
        return predictions