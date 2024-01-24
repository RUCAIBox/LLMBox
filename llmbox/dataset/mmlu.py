import numpy as np

from .multiple_choice_dataset import MultipleChoiceDataset

STEM_SUBJECTS = [
    'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science',
    'college_mathematics', 'college_physics', 'computer_security', 'conceptual_physics', 'electrical_engineering',
    'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'machine_learning'
]
HUMANITIES_SUBJECTS = [
    'formal_logic', 'high_school_european_history', 'high_school_us_history', 'high_school_world_history',
    'international_law', 'jurisprudence', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy',
    'prehistory', 'professional_law', 'world_religions'
]
SOCIAL_SCIENCES_SUBJECTS = [
    'econometrics', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics',
    'high_school_microeconomics', 'high_school_psychology', 'human_sexuality', 'professional_psychology',
    'public_relations', 'security_studies', 'sociology', 'us_foreign_policy'
]
OTHER_SUBJECTS = [
    'anatomy', 'business_ethics', 'clinical_knowledge', 'college_medicine', 'global_facts', 'human_aging', 'management',
    'marketing', 'medical_genetics', 'miscellaneous', 'nutrition', 'professional_accounting', 'professional_medicine',
    'virology'
]

MMLU_SUBJECTS = {
    'stem': STEM_SUBJECTS,
    'humanities': HUMANITIES_SUBJECTS,
    'social_sciences': SOCIAL_SCIENCES_SUBJECTS,
    'other': OTHER_SUBJECTS
}


class Mmlu(MultipleChoiceDataset):
    """The dataset of MMLU.

    Measuring Massive Multitask Language Understanding by Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt (ICLR 2021).

    Example:
        "question": "What is the embryological origin of the hyoid bone?",
        "choices": ["The first pharyngeal arch", "The first and second pharyngeal arches", "The second pharyngeal arch", "The second and third pharyngeal arches"],
        "answer": 3
    """

    instruction = "The following are multiple choice questions (with answers) about {}."
    evaluation_set = "test"
    example_set = "dev"
    load_args = ("cais/mmlu", "all")
    subject_column = "subject"

    def __init__(self, args, model, subset_name=None):
        self.instruction = self.instruction.format(subset_name)
        super().__init__(args, model, subset_name)

    def format_instance(self, instance):
        options = list(map(lambda op: " " + op, instance["choices"]))
        return dict(
            source="Question: " + instance["question"] + "\nAnswer:",
            target=options[instance["answer"]],
            options=options,
        )

    def calculate_metric(self, predictions):
        results, score_lists = super().calculate_metric(predictions)
        if "mmlu" in results:
            metric_entries = results["mmlu"].keys()
            for cat, cat_subjects in MMLU_SUBJECTS.items():
                cat_results = [results[f"mmlu:{subject}"] for subject in cat_subjects]
                results[f"mmlu[{cat}]"] = {m: np.mean([r[m] for r in cat_results]) for m in metric_entries}
        return results, score_lists

    @property
    def references(self):
        return [instance["answer"] for instance in self.evaluation_data]
