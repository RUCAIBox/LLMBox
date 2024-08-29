# Supported Datasets of LLMBox

We currently support 59+ commonly used datasets for LLMs.

## Understanding Evaluation Type

Each dataset is either a multiple-choice dataset or a generation dataset. You can find the difference between them at [here](https://github.com/RUCAIBox/LLMBox/tree/main/utilization#dataset-arguments)

## Understanding Subsets

Some datasets have multiple subsets. For example, Massive Multitask Language Understanding (`mmlu`) dataset contains 57 different subsets categorized into four categories: `stem`, `social_sciences`, `humanities`, and `other`.

While some other dataset is a subset of another dataset. For example, Choice Of Plausible Alternatives (`copa`) is a subset of `super_glue`.

See how to [load datasets with subsets](https://github.com/RUCAIBox/LLMBox/tree/main/docs/utilization/how-to-load-datasets-with-subsets.md).

## Understanding CoT

Some datasets support Chain-of-Thought reasoning. For example, Grade School Math 8K (`gsm8k`) supports three types of CoT: `base`, `least_to_most`, and `pal`.

## Supported Datasets

<table>
  <tr>
      <td><b>Dataset</b></td>
      <td><b>Subsets / Collections</b></td>
      <td><b>Evaluation Type</b></td>
      <td><b>CoT</b></td>
      <td><b>Notes</b></td>
  </tr>
  <tr>
      <td rowspan=3>AGIEval(<code>agieval</code>, alias of <code>agieval_single_choice</code> and <code>agieval_cot</code>)</td>
      <td><b>English</b>: <code>sat-en</code>, <code>sat-math</code>, <code>lsat-ar</code>, <code>lsat-lr</code>, <code>lsat-rc</code>, <code>logiqa-en</code>, <code>aqua-rat</code>, <code>sat-en-without-passage</code></td>
      <td rowspan=2>MultipleChoice</td>
      <td></td>
      <td rowspan=3></td>
  </tr>
  <tr>
      <td><code>gaokao-chinese</code>, <code>gaokao-geography</code>, <code>gaokao-history</code>, <code>gaokao-biology</code>, <code>gaokao-chemistry</code>, <code>gaokao-english</code>, <code>logiqa-zh</code></td>
      <td></td>
  </tr>
  <tr>
      <td><code>jec-qa-kd</code>, <code>jec-qa-ca</code>, <code>math</code>, <code>gaokao-physics</code>, <code>gaokao-mathcloze</code>, <code>gaokao-mathqa</code></td>
      <td>Generation</td>
      <td>✅</td>
  </tr>
  <tr>
      <td>Alpacal Eval (<code>alpaca_eval</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td>Single GPTEval</td>
  </tr>
  <tr>
      <td>Adversarial Natural Language Inference (<code>anli</code>)</td>
      <td><code>Round2</code> (default)</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>AI2's Reasoning Challenge (<code>arc</code>)</td>
      <td><code>ARC-Easy</code>, <code>ARC-Challenge</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td>Normalization</td>
  </tr>
  <tr>
      <td>BIG-Bench Hard (<code>bbh</code>)</td>
      <td><code>boolean_expressions</code>, ...</td>
      <td>Generation</td>
      <td>✅</td>
      <td></td>
  </tr>
  <tr>
      <td>Boolean Questions (<code>boolq</code>)</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>CommitmentBank (<code>cb</code>)</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td rowspan=4>C-Eval (<code>ceval</code>)</td>
      <td><b>stem</b>: <code>advanced_mathematics</code>, <code>college_chemistry</code>, ...</td>
      <td rowspan=4>MultipleChoice</td>
      <td rowspan=4></td>
      <td rowspan=4></td>
  </tr>
  <tr>
      <td><b>social science</b>: <code>business_administration</code>, <code>college_economics</code>, ...</td>
  </tr>
  <tr>
      <td><b>humanities</b>: <code>art_studies</code>, <code>chinese_language_and_literature</code>, ...</td>
  </tr>
  <tr>
      <td><b>other</b>: <code>accountant</code>, <code>basic_medicine</code>, ...</td>
  </tr>
  <tr>
      <td rowspan=4>Massive Multitask Language Understanding in Chinese (<code>cmmlu</code>)</td>
      <td><b>stem</b>: <code>anatomy</code>, <code>astronomy</code>, ...</td>
      <td rowspan=4>MultipleChoice</td>
      <td rowspan=4></td>
      <td rowspan=4></td>
  </tr>
  <tr>
      <td><b>social science</b>: <code>ancient_chinese</code>, <code>business_ethics</code>, ...</td>
  </tr>
  <tr>
      <td><b>humanities</b>: <code>arts</code>, <code>chinese_history</code>, ...</td>
  </tr>
  <tr>
      <td><b>other</b>: <code>agronomy</code>, <code>chinese_driving_rule</code>, ...</td>
  </tr>
  <tr>
      <td>CNN Dailymail (<code>cnn_dailymail</code>)</td>
      <td><code>3.0.0</code> (default), ...</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Reasoning About Colored Objects (<code>color_objects</code>)</td>
      <td><i>bigbench</i> (reasoning_about_colored_objects)</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Commonsense QA (<code>commonsenseqa</code>)</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Choice Of Plausible Alternatives (<code>copa</code>)</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Conversational Question Answering (<code>coqa</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td>Download: <a href="https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json">train</a>, <a href="https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json">dev</a></td>
  </tr>
  <tr>
      <td>CrowS-Pairs (<code>crows_pairs</code>)</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Discrete Reasoning Over the content of Paragraphs (<code>drop</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td rowspan=3>GAOKAO (<code>gaokao</code>)</td>
      <td><b>Chinese</b>: <code>2010-2022_Chinese_Modern_Lit</code>, <code>2010-2022_Chinese_Lang_and_Usage_MCQs</code></td>
      <td rowspan=3>Generation</td>
      <td rowspan=3></td>
      <td rowspan=3>Metric: Exam scoring</td>
  </tr>
  <tr>
      <td><b>English</b>: <code>2010-2022_English_Reading_Comp</code>, <code>2010-2022_English_Fill_in_Blanks</code>, ...</td>
  </tr>
  <tr>
      <td><code>2010-2022_Math_II_MCQs</code>, <code>2010-2022_Math_I_MCQs</code>, ...</td>
  </tr>
  <tr>
      <td>Google-Proof Q&A (<code>GPQA</code>)</td>
      <td><code>gpqa_main</code> (default), <code>gpqa_extended</code>, ...</td>
      <td>MultipleChoice</td>
      <td>✅</td>
      <td></td>
  </tr>
  <tr>
      <td>Grade School Math 8K (<code>gsm8k</code>)</td>
      <td><code>main</code> (default), <code>socratic</code></td>
      <td>Generation</td>
      <td>✅</td>
      <td>Code exec</td>
  </tr>
  <tr>
      <td>HaluEval(<code>halueval</code>)</td>
      <td><code>dialogue_samples</code>, <code>qa_samples</code>, <code>summarization_samples</code></td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>HellaSWAG (<code>hellaswag</code>)</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>HumanEval (<code>humaneval</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td>Pass@K</td>
  </tr>
  <tr>
      <td>Instruction-Following Evaluation (<code>ifeval</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td><a href="https://huggingface.co/datasets/imbue/code-comprehension">Imbue Code Comprehension</a> (<code>imbue_code</code>)</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td><a href="https://huggingface.co/datasets/imbue/high_quality_private_evaluations">Imbue High Quality Private Evaluations</a> (<code>imbue_private</code>)</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td><a href="https://huggingface.co/datasets/imbue/high_quality_public_evaluations">Imbue High Quality Public Evaluations</a> (<code>imbue_public</code>)</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>LAnguage Modeling Broadened to Account for Discourse Aspects (<code>lambada</code>)</td>
      <td><code>default</code> (default), <code>de</code>, ... (source: <i>EleutherAI/lambada_openai</i>)</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Mathematics Aptitude Test of Heuristics (<code>math</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td>✅</td>
      <td></td>
  </tr>
  <tr>
      <td>Mostly Basic Python Problems (<code>mbpp</code>)</td>
      <td><code>full</code> (default), <code>sanitized</code></td>
      <td>Generation</td>
      <td></td>
      <td>Pass@K</td>
  </tr>
  <tr>
      <td rowspan=4>Massive Multitask Language Understanding(<code>mmlu</code>)</td>
      <td><b>stem</b>: <code>abstract_algebra</code>, <code>astronomy</code>, ...</td>
      <td rowspan=4>MultipleChoice</td>
      <td rowspan=4></td>
      <td rowspan=4></td>
  </tr>
  <tr>
      <td><b>social_sciences</b>: <code>econometrics</code>, <code>high_school_geography</code>, ...</td>
  </tr>
  <tr>
      <td><b>humanities</b>: <code>formal_logic</code>, <code>high_school_european_history</code>, ...</td>
  </tr>
  <tr>
      <td><b>other</b>: <code>anatomy</code>, <code>business_ethics</code>, ...</td>
  </tr>
  <tr>
      <td>Multi-turn Benchmark (<code>mt_bench</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td>Multi-turn GPTEval</td>
  </tr>
  <tr>
      <td>Natural Questions(<code>nq</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>OpenBookQA (<code>openbookqa</code>)</td>
      <td><code>main</code> (default), <code>additional</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td>Normalization</td>
  </tr>
  <tr>
      <td>Penguins In A Table (<code>penguins_in_a_table</code>)</td>
      <td><i>bigbench</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Physical Interaction: Question Answering (<code>piqa</code>)</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Question Answering in Context (<code>quac</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>ReAding Comprehension (<code>race</code>)</td>
      <td><code>high</code>, <code>middle</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td>Normalization</td>
  </tr>
  <tr>
      <td>Real Toxicity Prompts (<code>real_toxicity_prompts</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td><a href="https://www.perspectiveapi.com/">Perspective</a> Toxicity</td>
  </tr>
  <tr>
      <td>Recognizing Textual Entailment (<code>rte</code>)</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Social Interaction QA (<code>siqa</code>)</td>
      <td>/</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Stanford Question Answering Dataset (<code>squad, squad_v2</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Story Cloze Test (<code>story_cloze</code>)</td>
      <td><code>2016</code> (default), <code>2018</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td><a href='http://goo.gl/forms/aQz39sdDrO'>Manually download</a></td>
  </tr>
  <tr>
      <td>TL;DR (<code>tldr</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>TriviaQA (<code>triviaqa</code>)</td>
      <td><code>rc.wikipedia.nocontext</code> (default), <code>rc</code>, <code>rc.nocontext</code>, ...</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>TruthfulQA (<code>truthfulqa_mc</code>)</td>
      <td><code>multiple_choice</code> (default), <code>generation</code> (not supported)</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Vicuna Bench (<code>vicuna_bench</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td>GPTEval</td>
  </tr>
  <tr>
      <td>WebQuestions (<code>webq</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Words in Context (<code>wic</code>)</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Winogender Schemas (<code>winogender</code>)</td>
      <td><code>main</code>, <code>gotcha</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td>Group by gender</td>
  </tr>
  <tr>
      <td>WSC273 (<code>winograd</code>)</td>
      <td><code>wsc273</code> (default), <code>wsc285</code></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>WinoGrande (<code>winogrande</code>)</td>
      <td><code>winogrande_debiased</code> (default), ...</td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Conference on Machine Translation (<code>wmt21, wmt19, ...</code>)</td>
      <td><code>en-ro</code>, <code>ro-en</code>, ...</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Winograd Schema Challenge (<code>wsc</code>)</td>
      <td><i>super_glue</i></td>
      <td>MultipleChoice</td>
      <td></td>
      <td></td>
  </tr>
  <tr>
      <td>Extreme Summarization (<code>xsum</code>)</td>
      <td>/</td>
      <td>Generation</td>
      <td></td>
      <td></td>
  </tr>

</table>
