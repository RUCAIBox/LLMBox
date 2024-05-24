WMT_DATASETS = ["wmt10", "wmt13", "wmt14", "wmt15", "wmt16", "wmt17", "wmt18", "wmt19", "wmt21"]

DEFAULT_VLLM_DATASETS = {
    "alpaca_eval", "bbh", "cnn_dailymail", "color_objects", "coqa", "drop", "gaokao", "gsm8k", "halueval", "humaneval",
    "ifeval", "lambada", "math", "mbpp", "mt_bench", "nq", "quac", "real_toxicity_prompts", "squad", "squad_v2", "tldr",
    "triviaqa", "vicuna_bench", "webq", "xsum"
} | set(WMT_DATASETS)

DATASET_ALIASES = {
    "agieval": ["agieval_single_choice", "agieval_cot"],  # try to use MultipleChoiceDataset first
    "squad_v2": ["squad"],
}

for wmt in WMT_DATASETS:
    DATASET_ALIASES[wmt] = [f"translation_dataset"]

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

BBH_NO_CHOICE = ['dyck_languages', 'multistep_arithmetic_two', 'object_counting', 'word_sorting']

BBH_LETTER_CHOICE = [
    'date_understanding', 'disambiguation_qa', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects',
    'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'penguins_in_a_table',
    'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks',
    'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_three_objects'
]

# 8 subsets reported as Agieval English in llama paper
AGIEVAL_ENGLISH_SUBJECTS = [
    'lsat-ar', 'lsat-lr', 'lsat-rc', 'logiqa-en', 'sat-math', 'sat-en', 'aqua-rat', 'sat-en-without-passage'
]

AGIEVAL_SUBJECTS = {"English": AGIEVAL_ENGLISH_SUBJECTS}

AGIEVAL_EN_QA_TASKS = AGIEVAL_ENGLISH_SUBJECTS + ['gaokao-english']

AGIEVAL_ZH_QA_TASKS = [
    'logiqa-zh', 'jec-qa-kd', 'jec-qa-ca', 'gaokao-chinese', 'gaokao-geography', 'gaokao-history', 'gaokao-biology',
    'gaokao-chemistry', 'gaokao-physics', 'gaokao-mathqa'
]

AGIEVAL_EN_CLOZE_TASKS = ['math']

AGIEVAL_ZH_CLOZE_TASKS = ['gaokao-mathcloze']

AGIEVAL_MULTI_ANSWERS_TASKS = ['jec-qa-kd', 'jec-qa-ca', 'gaokao-physics']

AGIEVAL_NO_LETTER_CHOICE_TASKS = AGIEVAL_EN_CLOZE_TASKS + AGIEVAL_ZH_CLOZE_TASKS

AGIEVAL_ZH_PROMPT_TASKS = AGIEVAL_ZH_CLOZE_TASKS + AGIEVAL_ZH_QA_TASKS

AGIEVAL_EN_PROMPT_TASKS = AGIEVAL_EN_CLOZE_TASKS + AGIEVAL_EN_QA_TASKS

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
    "chinese_history": "中国历史",
    "chinese_literature": "中国文学",
    "chinese_teacher_qualification": "中国教师资格",
    "clinical_knowledge": "临床知识",
    "college_actuarial_science": "大学精算学",
    "college_education": "大学教育学",
    "college_engineering_hydrology": "大学工程水文学",
    "college_law": "大学法律",
    "college_mathematics": "大学数学",
    "college_medical_statistics": "大学医学统计",
    "college_medicine": "大学医学",
    "computer_science": "计算机科学",
    "computer_security": "计算机安全",
    "conceptual_physics": "概念物理学",
    "construction_project_management": "建设工程管理",
    "economics": "经济学",
    "education": "教育学",
    "electrical_engineering": "电气工程",
    "elementary_chinese": "小学语文",
    "elementary_commonsense": "小学常识",
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
    "security_study": "安全研究",
    "sociology": "社会学",
    "sports_science": "体育学",
    "traditional_chinese_medicine": "中医中药",
    "virology": "病毒学",
    "world_history": "世界历史",
    "world_religions": "世界宗教",
}

CEVAL_TRANS = {
    "computer_network": "计算机网络",
    "operating_system": "操作系统",
    "computer_architecture": "计算机组成",
    "college_programming": "大学编程",
    "college_physics": "大学物理",
    "college_chemistry": "大学化学",
    "advanced_mathematics": "高等数学",
    "probability_and_statistics": "概率统计",
    "discrete_mathematics": "离散数学",
    "electrical_engineer": "注册电气工程师",
    "metrology_engineer": "注册计量师",
    "high_school_mathematics": "高中数学",
    "high_school_physics": "高中物理",
    "high_school_chemistry": "高中化学",
    "high_school_biology": "高中生物",
    "middle_school_mathematics": "初中数学",
    "middle_school_biology": "初中生物",
    "middle_school_physics": "初中物理",
    "middle_school_chemistry": "初中化学",
    "veterinary_medicine": "兽医学",
    "college_economics": "大学经济学",
    "business_administration": "工商管理",
    "marxism": "马克思主义基本原理",
    "mao_zedong_thought": "毛泽东思想和中国特色社会主义理论体系概论",
    "education_science": "教育学",
    "teacher_qualification": "教师资格",
    "high_school_politics": "高中政治",
    "high_school_geography": "高中地理",
    "middle_school_politics": "初中政治",
    "middle_school_geography": "初中地理",
    "modern_chinese_history": "近代史纲要",
    "ideological_and_moral_cultivation": "思想道德修养与法律基础",
    "logic": "逻辑学",
    "law": "法学",
    "chinese_language_and_literature": "中国语言文学",
    "art_studies": "艺术学",
    "professional_tour_guide": "导游资格",
    "legal_professional": "法律职业资格",
    "high_school_chinese": "高中语文",
    "high_school_history": "高中历史",
    "middle_school_history": "初中历史",
    "civil_servant": "公务员",
    "sports_science": "体育学",
    "plant_protection": "植物保护",
    "basic_medicine": "基础医学",
    "clinical_medicine": "临床医学",
    "urban_and_rural_planner": "注册城乡规划师",
    "accountant": "注册会计师",
    "fire_engineer": "注册消防工程师",
    "environmental_impact_assessment_engineer": "环境影响评价工程师",
    "tax_accountant": "税务师",
    "physician": "医师资格"
}

CEVAL_SUBJECTS = {
    'stem': [
        'advanced_mathematics', 'college_chemistry', 'college_physics', 'college_programming', 'computer_architecture',
        'computer_network', 'discrete_mathematics', 'electrical_engineer', 'high_school_biology',
        'high_school_chemistry', 'high_school_mathematics', 'high_school_physics', 'metrology_engineer',
        'middle_school_biology', 'middle_school_chemistry', 'middle_school_mathematics', 'middle_school_physics',
        'operating_system', 'probability_and_statistics', 'veterinary_medicine'
    ],
    'social science': [
        'business_administration', 'college_economics', 'education_science', 'high_school_geography',
        'high_school_politics', 'mao_zedong_thought', 'marxism', 'middle_school_geography', 'middle_school_politics',
        'teacher_qualification'
    ],
    'humanities': [
        'art_studies', 'chinese_language_and_literature', 'high_school_chinese', 'high_school_history',
        'ideological_and_moral_cultivation', 'law', 'legal_professional', 'logic', 'middle_school_history',
        'modern_chinese_history', 'professional_tour_guide'
    ],
    'other': [
        'accountant', 'basic_medicine', 'civil_servant', 'clinical_medicine',
        'environmental_impact_assessment_engineer', 'fire_engineer', 'physician', 'plant_protection', 'sports_science',
        'tax_accountant', 'urban_and_rural_planner'
    ]
}

# See Gaokao.extract_choice_answer for the details
GAOKAO_TASKS = {
    "2010-2022_Math_II_MCQs": "single_answer_mcq",
    "2010-2022_Math_I_MCQs": "single_answer_mcq",
    "2010-2022_History_MCQs": "single_answer_mcq",
    "2010-2022_Biology_MCQs": "single_answer_mcq",
    "2010-2022_Political_Science_MCQs": "single_answer_mcq",
    "2010-2022_Physics_MCQs": "multi_answers_mcq",
    "2010-2022_Chemistry_MCQs": "single_answer_mcq",
    "2010-2013_English_MCQs": "single_answer_mcq",
    "2010-2022_Chinese_Modern_Lit": "multi_mcqs",
    "2010-2022_English_Fill_in_Blanks": "multi_mcqs",
    "2012-2022_English_Cloze_Test": "seven_option",
    "2010-2022_Geography_MCQs": "multi_mcqs",
    "2010-2022_English_Reading_Comp": "multi_mcqs",
    "2010-2022_Chinese_Lang_and_Usage_MCQs": "multi_mcqs"
}

# The total score of each subsets. Each instance may have different score.
GAOKAO_CHINESE_TASKS_SCORE = {
    "2010-2022_Chinese_Modern_Lit": 261,
    "2010-2022_Chinese_Lang_and_Usage_MCQs": 240,
}

GAOKAO_ENGLISH_TASKS_SCORE = {
    "2010-2022_English_Reading_Comp": 940,
    "2010-2022_English_Fill_in_Blanks": 900,
    "2012-2022_English_Cloze_Test": 260,
    "2010-2013_English_MCQs": 105,
}

GAOKAO_TASKS_SCORE = dict(**GAOKAO_CHINESE_TASKS_SCORE, **GAOKAO_ENGLISH_TASKS_SCORE)
