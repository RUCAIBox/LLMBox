from logging import getLogger
from typing import List, Tuple

from ..metric import HaluEval
from .generation_dataset import GenerationDataset

logger = getLogger(__name__)


class Halueval_qa(GenerationDataset):
    """The dataset of HaluEval_qa.

    Example:
        question: Which magazine was started first Arthur's Magazine or First for Women?
        answer: First for Women was started first.
        hallucination: yes
    """

    instruction = ""
    example_set = "data"
    evaluation_set = "data"
    metrics = [HaluEval(type="qa")]
    load_args = ("pminervini/HaluEval", "qa_samples")
    extra_model_args = dict(temperature=0, stop='\n')

    def format_instance(self, instance):
        source = instruction_qa + "\n\n#Question#: " + instance["question"] + "\n#Answer#: " + instance[
            "answer"] + "\n#Your Judgement#:"
        return dict(
            source=source,
            target="",
        )

    @staticmethod
    def post_processing(predictions):
        new_predictions = []
        for pred in predictions:
            new_predictions.append(pred.strip().replace(".", ""))
        return new_predictions

    @property
    def references(self):
        return [instance["hallucination"].capitalize() for instance in self.evaluation_data]


instruction_qa = """I want you act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if the answer misunderstands the question context and intention.
#Question#: What is a rare breed of dog that was derived as a variant of Rat Terrier, Shiloh Shepherd dog or American Hairless Terrier?
#Answer#: American Hairless Terrier
#Your Judgement#: No

You are trying to determine if there is a factual contradiction between the answer and the world knowledge. Some information in the answer might be fabricated.
#Question#: Are the New Orleans Outfall Canals the same length as the Augusta Canal?
#Answer#: No, the New Orleans Outfall Canals and the Augusta Canal are not the same length. The Orleans Canal is approximately 3.6 miles (5.8 kilometers) long while the Augusta Canal is approximately 7 miles (11.3 kilometers) long.
#Your Judgement#: Yes
#Question#: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
#Answer#: U.S Highway 70
#Your Judgement#: Yes

You are trying to determine if the answer is too general or too specific to answer the question at an appropriate level of specificity.
#Question#: What genre do Superheaven and Oceansize belong to?
#Answer#: Superheaven and Oceansize belong to the rock genre.
#Your Judgement#: No
#Question#: What profession do Kōbō Abe and Agatha Christie share?
#Answer#: Playwright.
#Your Judgement#: No

You are trying to determine if the answer can be correctly inferred from the knowledge.
#Question#: Which band has more members, Muse or The Raconteurs?
#Answer#: Muse has more members than The Raconteurs.
#Your Judgement#: Yes
#Question#: Which is currently more valuable, Temagami-Lorrain Mine or Meadowbank Gold Mine?
#Answer#: Meadowbank Gold Mine, since Meadowbank Gold Mine is still producing gold and the TemagamiLorrain Mine has been inactive for years.
#Your Judgement#: No

You should try your best to determine if the answer contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be \"Yes\" or \"No\""."""
