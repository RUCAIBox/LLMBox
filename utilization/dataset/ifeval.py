from ..metric import IFEval
from .generation_dataset import GenerationDataset


class Ifeval(GenerationDataset):
    """The dataset of Ifeval.

    This dataset contains the prompts used in Google's Instruction-Following Evaluation for Large Language Models.

    Examples:
        prompt: Write a 300+ word summary of the wikipedia page "https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.
        instruction_id_list: ['punctuation:no_comma','detectable_format:number_highlighted_sections','length_constraints:number_words']
        kwargs:
           [{'num_highlights': None,
            'relation': None,
            'num_words': None,
            'num_placeholders': None,
            'prompt_to_repeat': None,
            'num_bullets': None,
            'section_spliter': None,
            'num_sections': None,
            'capital_relation': None,
            'capital_frequency': None,
            'keywords': None,
            'num_paragraphs': None,
            'language': None,
            'let_relation': None,
            'letter': None,
            'let_frequency': None,
            'end_phrase': None,
            'forbidden_words': None,
            'keyword': None,
            'frequency': None,
            'num_sentences': None,
            'postscript_marker': None,
            'first_word': None,
            'nth_paragraph': None},...]
    """

    instruction = "{source}"
    example_set = "train"
    evaluation_set = "train"
    load_args = ("HuggingFaceH4/ifeval",)
    metrics = [IFEval(type="strict"), IFEval(type="loose")]
    extra_model_args = dict(max_tokens=1024, temperature=0)

    def format_instance(self, instance):
        return dict(source=instance["prompt"], target="")

    @property
    def references(self):
        processed_evaluation = []
        for evaluation in self.evaluation_data:
            kwargs = evaluation["kwargs"]
            modified_list = [{k: v for k, v in d.items() if v is not None} for d in kwargs]
            evaluation["kwargs"] = modified_list
            processed_evaluation.append(evaluation)
        return processed_evaluation
