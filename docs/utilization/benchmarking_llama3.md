# Benchmarking LLaMA-3 with LLMBox

This tutorial demonstrates how to benchmark the LLaMA-3 model using LLMBox.

Meta has published its new [LLaMA-3](https://ai.meta.com/blog/meta-llama-3) pre-trained models and instruct models. The official model card includes a [benchmark results](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md) across 15 datasets under [differenct settings](https://github.com/meta-llama/llama3/blob/bb55334adcedfa9f5da66d2e1ed64e6f3dbd82ed/eval_details.md).

![Pre-trained model](https://web.archive.org/web/20240418195246im_/https://scontent-sjc3-1.xx.fbcdn.net/v/t39.2365-6/439014085_432870519293677_8138616034495713484_n.png?_nc_cat=1&ccb=1-7&_nc_sid=e280be&_nc_ohc=WeZO1csole8Ab4L36qo&_nc_ht=scontent-sjc3-1.xx&oh=00_AfDPzmw_mziuwCcdwjGAr5QjME5F9lnaPhV3BncYHgThrg&oe=663BA934)

![Instruct model](https://web.archive.org/web/20240418195246im_/https://scontent-sjc3-1.xx.fbcdn.net/v/t39.2365-6/438037375_405784438908376_6082258861354187544_n.png?_nc_cat=1&ccb=1-7&_nc_sid=e280be&_nc_ohc=upKu3iJBkXUAb50aAlY&_nc_ht=scontent-sjc3-1.xx&oh=00_AfD1np8Pj9ty_-sxPnhP3esNVlsvPUPNBNezjHUWqtggYw&oe=663BA20A)

For all the evaluations, meta uses their internal evaluations library.

Here, we detail how you can reproduce these results using LLMBox in the same setup.

## Step 1: Install LLMBox

Begin by installing LLMBox. Execute the following commands:

```bash
git clone https://github.com/RUCAIBox/LLMBox.git && cd LLMBox
pip install -r requirements.txt
```

Alternatively, utilize the faster Python package installer [`uv`](https://github.com/astral-sh/uv) to expedite the `pip install` process.

LLMBox is a comprehensive library for implementing LLMs, including **a unified training pipeline** and **comprehensive model evaluation**. For our purpose, we will use the evaluation component (`inference.py` and `utilization`) of LLMBox to conduct our benchmarks.

## Step 2: Download the LLaMA-3 Model

Visit the Huggingface model repository ([`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) for example) and follow the instructions to download the LLaMA-3 model in Huggingface format.

After accepting the terms, utilize the [Huggingface CLI](https://huggingface.co/docs/huggingface_hub/v0.21.4/guides/cli#getting-started) to log into your account (`huggingface-cli login`). Then, with the same account, you can manually download the model or load directly from Huggingface Hub.

For manual download, you can enable the [`hf-transfer`](https://huggingface.co/docs/huggingface_hub/v0.21.4/guides/download#faster-downloads) to accelerate. The original model weights can also be excluded from download:

```bash
mkdir meta-llama
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
 --resume-download meta-llama/Meta-Llama-3-8B \
 --local-dir meta-llama/Meta-Llama-3-8B \
 --exclude "original/*" \
 --local-dir-use-symlinks False
```

## Step 3: Benchmark LLaMA-3 Base Pre-trained Model with LLMBox

TOC:

- [GenerationDataset with few-shot settings: SQuAD v2, QuAC, DROP, and TriviaQA](#generationdataset-with-few-shot-settings)
- [Perplexity over the Options: BoolQ, CommonSenseQA, and WinoGrande](#perplexity-over-the-options)
- [Probability over Choice Characters: MMLU and ARC-Challenge](#probability-over-choice-characters)
- [AGIEval English](#agieval-english)
- [Chain-of-Thought strategy: BIG-Bench Hard](#chain-of-thought-strategy)

Then we can benchmark the LLaMA-3 base pretrained model on 11 datasets with different settings.

By default, LLMBox will use all visible GPUs. The `CUDA_VISIBLE_DEVICES` environment variable or the `--cuda` flag allows you to specify particular GPUs to utilize.

View the full test scripts at [here](https://github.com/RUCAIBox/LLMBox/tree/main/docs/examples/benchmarking_llama3.sh).

### GenerationDataset with few-shot settings

To streamline the process, evaluate multiple datasets with a similar setup concurrently by grouping them by the number of shots:

```bash
python inference.py \
  --model meta-llama/Meta-Llama-3-8B \
  --dataset squad_v2 quac \
  --num_shots 1 \
  --max_example_tokens 4096
```

This command will evaluate the model on the `SQuAD v2` and `QuAC` datasets with 1-shot setting. Since LLaMA-3 has a context lenght of 8192, we can set the maximal length of examples to 4096.

> [!NOTE]
> For efficiency, we enabled vllm for `GenerationDataset` by default. You can disable it by setting `--vllm False`.

The DROP dataset is a challenging dataset used to evaluate the capabilities of natural language processing models in discrete reasoning and text comprehension: given a passage and a question, the model is required to accurately answer the question based on the information in the passage.

DROP (3-shots) randomly draws 3 examples from `example_set = "train"` for each instance:

```bash
python inference.py \
  --model meta-llama/Meta-Llama-3-8B \
  --dataset drop \
  --num_shots 3
```

The DROP dataset uses two evaluation metrics:

1. EM (Exact Match): Calculates the proportion of predictions that match the ground truth answer exactly. This metric measures the accuracy of the model, awarding points only when the model's prediction exactly matches the standard answer.

2. F1 Score: Considers both precision and recall. For the model's predicted answer and the ground truth answer, the F1 score calculates the overlap between them, which is particularly useful for answers that contain multiple words. The F1 score is more lenient, allowing partially correct answers to receive some credit.

Meta reports F1 scores for the DROP dataset.

In LLMBox, TriviaQA by default uses the same subset as LLaMA-3, i.e. the WIKI validation set `rc.wikipedia.nocontext`.

```bash
python inference.py \
  --model meta-llama/Meta-Llama-3-8B \
  --dataset triviaqa \
  --num_shots 5
```

### Perplexity over the Options

For WinoGrande, LLaMA-3 uses a choice based setup for evaluation. The missing blank is filled with the two possible choices and then compute log-likelihood over the suffix. LLMBox also implements WinoGrande as a `MultipleChoiceDataset`.

> [!TIP]
> Currently vllm (v0.5.0) does not support returning the logprobs with prefix caching enabled. As a workaround, we enable our implementation of prefix caching by default for `MultipleChoiceDataset`. You need to specify a batch size for the model with `--batch_size` or `-b` flag.

```bash
python inference.py \
  --model meta-llama/Meta-Llama-3-8B \
  --batch_size 128:auto \
  --dataset winogrande \
  --num_shots 5
```

LLMBox uses `ppl_no_option` ranking type by default, where one instance is splitted into sub-instances by the choices. In each sub-instance, only the corresponding choice is provided and we calculate the perplexity over it. We choose the option with the lowest perplexity as the answer.

<table>
    <tr>
        <td><b>Dataset</b></td>
        <td><b>Evaluation Method</b></td>
        <td><b>Prompt</b></td>
    </tr>
    <tr>
        <td rowspan=4><p><b>MultipleChoice</b></p>
<pre><code>{
  "question":
    "What is the ...?",
  "choices": [
    "The first",
    "The second",
    ...
  ],
  "answer": 3
}</code></pre></td>
        <td rowspan=2><p><code>get_ppl</code></p><p>Calculate perplexity of the option text based on the source text</p></td>
        <td><p style="text-align: center;"><code>ppl_no_option</code></p>
<p><pre><code>Q: What is ...?
A: The first
   └--ppl--┘</code></pre></p></td>
    </tr>
    <tr>
        <td><p style="text-align: center;"><code>ppl</code></p>
<p><pre><code>Q: What is ...?
A. The first
B. The second
C. ...
A: A. The first
   └----ppl---┘</code></pre></p></td>
    </tr>
    <tr>
        <td><p><code>get_prob</code></p><p>Get the probability of each option label</p></td>
        <td><p style="text-align: center;"><code>prob</code></p>
<p><pre><code>Q: What is ...?
A. The first
B. The second
C. ...
A: _
   └→ [A B C D]</code></pre></p></td>
    </tr>
    <tr>
        <td><p><code>generation</code></p><p>Generate based on the source text</p></td>
        <td><p style="text-align: center;"><code>prob</code></p>
<p><pre><code>Q: What is ...?
A. The first
B. The second
C. ...
A: _ (max_tokens=1)</code></pre></p></td>
    </tr>
</table>

BoolQ (0-shot) dataset uses the same settings as LLaMA-1 and LLaMA-2:

```bash
python inference.py \
  --model meta-llama/Meta-Llama-3-8B \
  --batch_size 128:auto \
  --dataset boolq  # --ranking_type ppl_no_option
```

CommonSenseQA (7-shots):

```bash
python inference.py \
  --model meta-llama/Meta-Llama-3-8B \
  --batch_size 128:auto \
  --dataset commonsenseqa \
  --num_shots 7  # --ranking_type ppl_no_option
```

Notes: CoT currently not supported in CSQA.

### Probability over Choice Characters

For ARC-Challenge, LLaMA-3 uses the MMLU setup for evaluation where they provide all the choices in the prompt and calculate likelihood over choice characters (A, B, C, D, ...).

In LLMBox, you can use the `--ranking_type prob` option to calculate in the same way. You can see the difference between it and other ranking types in our [documentation](https://github.com/RUCAIBox/LLMBox/tree/main/utilization#dataset-arguments).

MMLU (5-shot):

```bash
python inference.py \
  --model meta-llama/Meta-Llama-3-8B \
  --batch_size 128:auto \
  --dataset mmlu \
  --num_shots 5 \
  --ranking_type prob \
  --max_example_tokens 4096
```

ARC-Challenge (25-shot):

```bash
python inference.py \
  --model meta-llama/Meta-Llama-3-8B \
  --batch_size 32:auto \
  --dataset arc:ARC-Challenge \
  --num_shots 25 \
  --ranking_type prob \
  --max_example_tokens 4096
```

### AGIEval English

Then we can evaluate on the `English` subcollection (includes 8 English subsets) of `AGIEval` dataset following the [default settings](https://github.com/ruixiangcui/AGIEval).

Notes that the `--num_shots` option is used to specify the maximum number of few-shots examples, and for subsets with fewer examples (like some subsets in AGIEval), the actual number of shots will be less than the specified number (3-5 shots in the case of AGIEval).

```bash
python inference.py \
  --model meta-llama/Meta-Llama-3-8B \
  --dataset agieval:[English] \
  --num_shots 5 \
  --batch_size 16:auto \
  --max_example_tokens 2560
```

You might see the following warning message:

```txt
2024-04-20 21:54:41 WARNING The example data of agieval:lsat-rc only has 3 instances, but the few-shot number is set to 5. Setting the few-shot number to 3.
2024-04-20 21:54:41 WARNING The example data of agieval:lsat-lr only has 3 instances, but the few-shot number is set to 5. Setting the few-shot number to 3.
...
2024-04-20 21:56:21 INFO Evaluating get_ppl on agieval 8 subsets (model_attr={'type': 'base', 'model_backend': 'huggingface', 'model_max_input': 8191, 'model_max_input_and_output': 8192}, ppl_kwargs=None, num_shots=3-5, len=11447, num_instances=2546, use_cache=False)
```

### Chain-of-Thought strategy

LLaMA-3 uses the Chain-of-Thought strategy in BIG-Bench Hard (BBH) dataset. In LLMBox, you can use the `--cot base` option to evaluate in the same way.

```bash
python inference.py \
  --model meta-llama/Meta-Llama-3-8B \
  --dataset bbh \
  --cot base \
  --num_shots 3 \
  --vllm
```

Then we report the Exact Match (EM) scores for it.

## Step 4: Benchmark LLaMA-3 Instruct Model with LLMBox

TOC:

- [LLaMA-3 Instruct model: MMLU and GPQA](#llama-3-instruct-model)
- [Pass@K Metrics: HumanEval](#passk-metrics)
- [Self-consistency: MATH and GSM8K](#self-consistency)

The LLaMA-3 Instruct model is evaluated on 5 datasets with different settings.

### LLaMA-3 Instruct model

In LLMBox, you can use `--model_type` to specify the model type. For an instruction tuned model, you can use `--model_type chat`. If model_type is not provided, LLMBox will automatically detect from the model name.

LLMBox also loads the chat_template automatically. You can specify the `--chat_template` option to use a custom chat template.

MMLU (5-shots):

```bash
python inference.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --batch_size 128:auto \
  --dataset mmlu \
  --max_example_tokens 4096 \
  --ranking_type prob \
  --num_shots 5
```

For GPQA (0-shot), we use the default settings:

```bash
python inference.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset gpqa
```

You can also see [gpqa](https://github.com/RUCAIBox/LLMBox/tree/main/docs/utilization/how-to-load-dataset-gpqa.md) for additional settings.

### Pass@K Metrics

For HumanEval dataset, LLaMA-3 uses the Pass@1 metrics for evaluation.

```bash
python inference.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset humaneval \
  --pass_at_k 1 \
  --sample_num 20 \
  --temperature 0.1 \
  --max_example_tokens 2560
```

The core idea of Pass@K is the probability that among the $k$ code outputs generated by a computational model for a single problem input, at least one code can pass the verification. However, directly calculating this metric requires repeatedly testing a single problem and generating $k$ codes in each test, leading to high computational costs. To reduce the computational complexity of the evaluation, an unbiased estimation approach is commonly used to approximate the value of Pass@K. The estimation formula is as follows:

$$\text{Pass@$k$} = \mathbb{E}\left(1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right)$$

where $n$ represents the number of codes synthesized for each test problem, and $c$ represents the number of codes that pass the test. According to the law of large numbers, as the total sample size n increases, the accuracy of this estimation also improves.

We evaluate the `pass@1` metric by setting the `--sample_num` parameter to synthesize $20$ solutions for each test problem. The temperature should be set to greater than $0$ to ensure that different answers can be generated for the same problem.

### Self-consistency

Both MATH and GSM8K use `maj@1` metric (majority at 1) for evaluation, which is equivalent to the exact match (EM) scores with self-consistency `--majority 1` option.

For GSM8K (8-shots, CoT), we use [PAL](https://reasonwithpal.com/) CoT strategy:

```bash
python inference.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset gsm8k \
  --num_shots 8 \
  --max_example_tokens 2560 \
  --cot pal
```

MATH (4-shots, CoT):

```bash
python inference.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset math \
  --num_shots 4 \
  --max_example_tokens 2560 \
  --cot base
```

You can further the CoT strategy by adjusting the instruction template, which in MATH datasets is `--instruction "Solve the following math problem.\n\nQ: {problem}\nA:"`.

## Step 5: Report the Results

We are done with the evaluation now. You can collect the results and report them. Check the `logs` directory for the logs of each evaluation and `evaluation_results` directory for model raw inputs and outputs for each evaluation instance.

View the full test scripts at [here](https://github.com/RUCAIBox/LLMBox/tree/main/docs/examples/benchmarking_llama3.sh).

### LLaMA-3 (8B) Pre-trained Model

<table>
  <tr>
   <td><strong>Category</strong>
   </td>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Meta Reported</strong>
   </td>
   <td><strong>LLMBox Measured</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="6" >General
   </td>
   <td>MMLU (5-shot)
   </td>
   <td>66.6
   </td>
   <td>65.76
   </td>
  </tr>
  <tr>
   <td>AGIEval English (3-5 shot)
   </td>
   <td>45.9
   </td>
   <td>33.3
   </td>
  </tr>
  <tr>
   <td>CommonSenseQA (7-shot)
   </td>
   <td>72.6
   </td>
   <td>73.0
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>76.1
   </td>
   <td>77.8
   </td>
  </tr>
  <tr>
   <td>BIG-Bench Hard (3-shot, CoT)
   </td>
   <td>61.1
   </td>
   <td>53.0
   </td>
  </tr>
  <tr>
   <td>ARC-Challenge (25-shot)
   </td>
   <td>78.6
   </td>
   <td>78.7
   </td>
  </tr>
  <tr>
   <td>Knowledge reasoning
   </td>
   <td>TriviaQA-Wiki (5-shot)
   </td>
   <td>78.5
   </td>
   <td>75.4
   </td>
  </tr>
  <tr>
   <td rowspan="4" >Reading comprehension
   </td>
   <td>SQuAD (1-shot, EM)
   </td>
   <td>76.4
   </td>
   <td>57.1
   </td>
  </tr>
  <tr>
   <td>QuAC (1-shot, F1)
   </td>
   <td>44.4
   </td>
   <td>37.8
   </td>
  </tr>
  <tr>
   <td>BoolQ (0-shot)
   </td>
   <td>75.7
   </td>
   <td>82.7
   </td>
  </tr>
  <tr>
   <td>DROP (3-shot, F1)
   </td>
   <td>58.4
   </td>
   <td>57.3
   </td>
  </tr>
</table>

### LLaMA-3 (8B) Instruct Model

<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Meta Reported</strong>
   </td>
   <td><strong>LLMBox Measured</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>68.4
   </td>
   <td>66.6
   </td>
  </tr>
  <tr>
   <td>GPQA (0-shot)
   </td>
   <td>34.2
   </td>
   <td>29.1
   </td>
  </tr>
  <tr>
   <td>HumanEval (0-shot)
   </td>
   <td>62.2
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>GSM-8K (8-shot, CoT)
   </td>
   <td>79.6
   </td>
   <td>73.7
   </td>
  </tr>
  <tr>
   <td>MATH (4-shot, CoT)
   </td>
   <td>30.0
   </td>
   <td>12.9
   </td>
  </tr>
</table>