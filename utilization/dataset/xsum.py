from functools import cached_property

from ..metric import Rouge
from .generation_dataset import GenerationDataset


class Xsum(GenerationDataset):
    """The dataset of Xsum.

    The Extreme Summarization (XSum) dataset is a dataset for evaluation of abstractive single-document summarization systems.

    Examples:
        document: The move is in response to an £8m cut in the subsidy received from the Department of Employment and Learning (DEL).\nThe cut in undergraduate places will come into effect from September 2015.\nJob losses will be among both academic and non-academic staff and Queen\'s says no compulsory redundancies should be required.\nThere are currently around 17,000 full-time undergraduate and postgraduate students at the university, and around 3,800 staff.\nQueen\'s has a current intake of around 4,500 undergraduates per year.\nThe university aims to reduce the number of student places by 1,010 over the next three years.\nThe BBC understands that there are no immediate plans to close departments or courses, but that the cuts in funding may put some departments and courses at risk.\nThe Education Minister Stephen Farry said he recognised that some students might now choose to study in other areas of the UK because of the cuts facing Northern Ireland\'s universities.\n"Some people will now be forced to look to opportunities in other parts of Great Britain and may not return to our economy," he said.\n"Defunding our investment in skills, particularly at a time when we\'re trying to grow the economy does not make a lot of sense. What\'s happening is we\'re going backwards.\n"The loss of any place is damaging to our economy, all subjects teach our young people critical skills."\nQueen\'s vice-chancellor Patrick Johnston said the cuts had the potential to damage the reputation of the university.\n"The potential negative impact, not just on the university but on the local economy is very significant," he said.\n"It\'s the last thing we want to do, but we have to begin to focus on those areas where we can grow the organisation and develop it - it\'s clear we can no longer depend on the public purse to fund tuition.\n"If we\'re not competitive we will not attract the best students, and we will not attract the best staff."\nJust under £100m, a third of the university\'s income, comes from the Northern Ireland Executive.\nDEL\'s budget was reduced by £62m earlier this year, and its budget for higher education institutions fell from £203m to £186m, a reduction of 8.2%.\nUlster University announced in February that it was dropping 53 courses.\nIt will be cutting jobs and student places, but it has not yet revealed how many.
        summary: Queen's University Belfast is cutting 236 jobs and 290 student places due to a funding reduction.
    """

    instruction = "{document}\n\nTL;DR:"
    evaluation_set = "train"
    example_set = None
    metrics = [Rouge()]
    load_args = ("EdinburghNLP/xsum",)
    extra_model_args = dict(temperature=0)

    def format_instance(self, instance):
        instance["target"] = instance["summary"]
        return instance

    @cached_property
    def references(self):
        return [instance["summary"] for instance in self.evaluation_data]
