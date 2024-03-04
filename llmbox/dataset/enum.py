MMLU_STEM_SUBJECTS = [
    'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science',
    'college_mathematics', 'college_physics', 'computer_security', 'conceptual_physics', 'electrical_engineering',
    'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'machine_learning'
]
MMLU_HUMANITIES_SUBJECTS = [
    'formal_logic', 'high_school_european_history', 'high_school_us_history', 'high_school_world_history',
    'international_law', 'jurisprudence', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy',
    'prehistory', 'professional_law', 'world_religions'
]
MMLU_SOCIAL_SCIENCES_SUBJECTS = [
    'econometrics', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics',
    'high_school_microeconomics', 'high_school_psychology', 'human_sexuality', 'professional_psychology',
    'public_relations', 'security_studies', 'sociology', 'us_foreign_policy'
]
MMLU_OTHER_SUBJECTS = [
    'anatomy', 'business_ethics', 'clinical_knowledge', 'college_medicine', 'global_facts', 'human_aging', 'management',
    'marketing', 'medical_genetics', 'miscellaneous', 'nutrition', 'professional_accounting', 'professional_medicine',
    'virology'
]

MMLU_SUBJECTS = {
    'stem': MMLU_STEM_SUBJECTS,
    'humanities': MMLU_HUMANITIES_SUBJECTS,
    'social_sciences': MMLU_SOCIAL_SCIENCES_SUBJECTS,
    'other': MMLU_OTHER_SUBJECTS
}

BBH_PROMPTS = {
    'boolean_expressions': "Evaluate the result of a random Boolean expression.",
    'causal_judgement': "Answer questions about causal attribution.",
    'date_understanding': "Infer the date from context.",
    'disambiguation_qa': "Clarify the meaning of sentences with ambiguous pronouns.",
    'dyck_languages': "Correctly close a Dyck-n word.",
    'formal_fallacies': "Distinguish deductively valid arguments from formal fallacies.",
    'geometric_shapes': "Name geometric shapes from their SVG paths.",
    'hyperbaton': "Order adjectives correctly in English sentences.",
    'logical_deduction_five_objects': "A logical deduction task which requires deducing the order of a sequence of objects.",
    'logical_deduction_seven_objects': "A logical deduction task which requires deducing the order of a sequence of objects.",
    'logical_deduction_three_objects': "A logical deduction task which requires deducing the order of a sequence of objects.",
    'movie_recommendation': "Recommend movies similar to the given list of movies.",
    'multistep_arithmetic_two': "Solve multi-step arithmetic problems.",
    'navigate': "Given a series of navigation instructions, determine whether one would end up back at the starting point.",
    'object_counting': "Questions that involve enumerating objects and asking the model to count them.",
    'penguins_in_a_table': "Answer questions about a table of penguins and their attributes.",
    'reasoning_about_colored_objects': "Answer extremely simple questions about the colors of objects on a surface.",
    'ruin_names': "Select the humorous edit that 'ruins' the input movie or musical artist name.",
    'salient_translation_error_detection': "Detect the type of error in an English translation of a German source sentence.",
    'snarks': "Determine which of two sentences is sarcastic.",
    'sports_understanding': "Determine whether an artificially constructed sentence relating to sports is plausible or not.",
    'temporal_sequences': "Task description: Answer questions about which times certain events could have occurred.",
    'tracking_shuffled_objects_five_objects': "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.",
    'tracking_shuffled_objects_seven_objects': "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.",
    'tracking_shuffled_objects_three_objects': "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.",
    'web_of_lies': "Evaluate a random boolean function expressed as a word problem.",
    'word_sorting': "Sort a list of words."
}

BBH_NO_CHOICE = [
    'dyck_languages',
    'multistep_arithmetic_two',
    'object_counting',
    'word_sorting'
]

BBH_LETTER_CHOICE = [
    'date_understanding',
    'disambiguation_qa',
    'geometric_shapes',
    'hyperbaton',
    'logical_deduction_five_objects',
    'logical_deduction_seven_objects',
    'logical_deduction_three_objects',
    'movie_recommendation',
    'penguins_in_a_table',
    'reasoning_about_colored_objects',
    'ruin_names',
    'salient_translation_error_detection',
    'snarks',
    'temporal_sequences',
    'tracking_shuffled_objects_five_objects',
    'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_three_objects'
]

AGIEVAL_WORDS = [
    ["问题：", "Q: "],
    ["选项：", "Answer Choices: "],
    ["答案：从A到{}, 我们应选择", "A: Among A through {}, the answer is"],
    ["从A到{}, 我们应选择什么？让我们逐步思考：", "Let's think step by step."],
    ["问题的解析:", "Explanation for Problem:"],
    ["答案是", "The answer is therefore"],
    ["问题. ", "Problem. "],
    ["从以下选项中选择: ", "Choose from the following options: "],
    ["答案：", "A: The answer is"],
    ["答案：让我们逐步思考：", "A: Let's think step by step."]
]

AGIEVAL_EN_QA = [
    'lsat-ar', 'lsat-lr', 'lsat-rc',
    'logiqa-en', 'sat-math', 'sat-en',
    'aqua-rat', 'sat-en-without-passage', 'gaokao-english'
]

AGIEVAL_ZH_QA = [
    'logiqa-zh', 'jec-qa-kd', 'jec-qa-ca', 'gaokao-chinese',
    'gaokao-geography', 'gaokao-history', 'gaokao-biology',
    'gaokao-chemistry', 'gaokao-physics', 'gaokao-mathqa'
]

AGIEVAL_EN_CLOZE = ['math']

AGIEVAL_ZH_CLOZE = ['gaokao-mathcloze']

AGIEVAL_MULTI_CHOICE = ['jec-qa-kd', 'jec-qa-ca', 'gaokao-physics']

AGIEVAL_CHINESE_TASK = AGIEVAL_ZH_CLOZE + AGIEVAL_ZH_QA

AGIEVAL_NO_LETTER_CHOICE = AGIEVAL_EN_CLOZE + AGIEVAL_ZH_CLOZE

# CMMLU: copied from https://github.com/haonan-li/CMMLU/blob/master/src/categories.py

CMMLU_NAME_TRANS = {
    "agronomy": "农学",
    "anatomy": "解剖学",
    "ancient_chinese": "古汉语",
    "arts": "艺术学",
    "astronomy": "天文学",
    "business_ethics": "商业伦理",
    "chinese_civil_service_exam": "中国公务员考试",
    "chinese_driving_rule": "中国驾驶规则",
    "chinese_food_culture": "中国饮食文化",
    "chinese_foreign_policy": "中国外交政策",
    "chinese_history":"中国历史",
    "chinese_literature": "中国文学",
    "chinese_teacher_qualification": "中国教师资格",
    "clinical_knowledge": "临床知识",
    "college_actuarial_science":"大学精算学",
    "college_education":"大学教育学",
    "college_engineering_hydrology": "大学工程水文学",
    "college_law": "大学法律",
    "college_mathematics": "大学数学",
    "college_medical_statistics":"大学医学统计",
    "college_medicine": "大学医学",
    "computer_science": "计算机科学",
    "computer_security": "计算机安全",
    "conceptual_physics": "概念物理学",
    "construction_project_management": "建设工程管理",
    "economics": "经济学",
    "education": "教育学",
    "electrical_engineering": "电气工程",
    "elementary_chinese":"小学语文",
    "elementary_commonsense":"小学常识",
    "elementary_information_and_technology": "小学信息技术",
    "elementary_mathematics": "初等数学",
    "ethnology": "民族学",
    "food_science": "食品科学",
    "genetics": "遗传学",
    "global_facts": "全球事实",
    "high_school_biology": "高中生物",
    "high_school_chemistry": "高中化学",
    "high_school_geography": "高中地理",
    "high_school_mathematics": "高中数学",
    "high_school_physics": "高中物理学",
    "high_school_politics": "高中政治",
    "human_sexuality": "人类性行为",
    "international_law": "国际法学",
    "journalism": "新闻学",
    "jurisprudence": "法理学",
    "legal_and_moral_basis": "法律与道德基础",
    "logical": "逻辑学",
    "machine_learning": "机器学习",
    "management": "管理学",
    "marketing": "市场营销",
    "marxist_theory": "马克思主义理论",
    "modern_chinese": "现代汉语",
    "nutrition": "营养学",
    "philosophy": "哲学",
    "professional_accounting": "专业会计",
    "professional_law": "专业法学",
    "professional_medicine": "专业医学",
    "professional_psychology": "专业心理学",
    "public_relations": "公共关系",
    "security_study":"安全研究",
    "sociology": "社会学",
    "sports_science": "体育学",
    "traditional_chinese_medicine": "中医中药",
    "virology": "病毒学",
    "world_history":"世界历史",
    "world_religions": "世界宗教",
}
