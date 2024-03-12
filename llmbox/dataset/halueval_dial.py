from logging import getLogger
from typing import List, Tuple

from ..metric import HaluEval
from .generation_dataset import GenerationDataset

logger = getLogger(__name__)


class Halueval_dial(GenerationDataset):
    """The dataset of Halueval.

    Example:
        dialogue_history:[Human]: Do you like Iron Man [Assistant]: Sure do! Robert Downey Jr. is a favorite. [Human]: Yes i like him too did you know he also was in Zodiac a crime fiction film.
        response:I'm not a fan of crime movies, but I did know that RDJ starred in Zodiac with Tom Hanks.
        hallucination:yes
    """

    instruction = ""
    example_set = "data"
    evaluation_set = "data"
    metrics = [HaluEval(type="dial")]
    load_args = ("pminervini/HaluEval", "dialogue_samples")
    extra_model_args = dict(temperature=0, stop='\n')

    def format_instance(self, instance):
        source = instruction_dial + "\n\n#Dialogue History#: " + instance[
            "dialogue_history"] + "\n#Response#: " + instance["response"] + "\n#Your Judgement#:"
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


instruction_dial = """I want you act as a response judge. Given a dialogue history and a response, your objective is to determine if the provided response contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if the true entity in the response is replaced with a highly similar entity.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Steven Spielberg was the director. He also directed insomnia and inception.
#Your Judgement#: Yes

You are trying to determine if the true entity in the response is replaced with a dissimilar entity.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Batman Begins was the director. He also directed insomnia and inception.
#Your Judgement#: Yes

You are trying to determine if the true entity in the response is replaced with a dissimilar entity in a different entity type.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: United States of America was the director. He also directed insomnia and inception.
#Your Judgement#: Yes

You should try your best to determine if the response contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be \"Yes\" or \"No\""."""
