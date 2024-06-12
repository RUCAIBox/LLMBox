import logging
import warnings
from dataclasses import dataclass
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers.hf_argparser import HfArg
from datasets import load_dataset
import torch
from accelerate.utils import set_seed
import random

@dataclass
class Arguments(TrainingArguments):
    model_name_or_path: str = HfArg(
        default=None,
        help="The model name or path, e.g., `meta-llama/Llama-2-7b-hf` or `./output/saved_model"
    )

    bits: int = HfArg(
        default=8,
        help="Bit size for quantization.",
        metadata={"choices": [2, 4, 8]}
    )

    group_size: int = HfArg(
        default=128,
        help="Group size for quantization. It is recommended to set the value to 128."
    )

    desc_act: bool = HfArg(
        default=True,
        help="Whether to use desc_act in quantization. Set to False can significantly speed up inference but the perplexity may slightly bad."
    )

    damp_percent: float = HfArg(
        default=0.1,
        help="Damping percentage for quantization."
    )

    num_samples: int = HfArg(
        default=128,
        help="Number of dataset samples to use."
    )

    seq_len: int = HfArg(
        default=512,
        help="Model sequence length."
    )

    use_triton: bool = HfArg(
        default=False,
        help="Whether to use triton in quantization."
    )

    batch_size: int = HfArg(
        default=1,
        help="Quantize batch size for processing dataset samples."
    )

    cache_examples_on_gpu: bool = HfArg(
        default=True,
        help="Whether to cache examples on GPU."
    )

    use_fast: bool = HfArg(
        default=True,
        help="Whether to use fast tokenizer."
    )

    trust_remote_code: bool = HfArg(
        default=False,
        help="Whether to trust remote code."
    )

    unquantized_model_dtype: str = HfArg(
        default="float16",
        help="which dtype to load the unquantised model.",
        metadata={"choices": ['float16', 'float32', 'bfloat16']}
    )



def get_c4(num_samples, tokenizer, seqlen):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', revision="607bd4c8450a42878aa9ddc051a65a055450ef87"
        )
    # If loading from Hugging Face encounters the ExpectedMoreSplits error, switch to local loading:
    # traindata = load_dataset('json', data_files='data/c4-train.00000-of-01024.json', split='train')
    trainloader = []
    for _ in range(num_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt', truncation=True, max_length=seqlen)
            if trainenc.input_ids.shape[1] >= seqlen:
                start = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
                end = start + seqlen
                inp = trainenc.input_ids[:, start:end]
                attention_mask = torch.ones_like(inp)
                trainloader.append({'input_ids': inp, 'attention_mask': attention_mask})
                break
    return trainloader

def quantize_and_save(args):
    if args.unquantized_model_dtype == 'float16':
        torch_dtype  = torch.float16
    elif args.unquantized_model_dtype == 'float32':
        torch_dtype  = torch.float32
    elif args.unquantized_model_dtype == 'bfloat16':
        torch_dtype  = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {args.unquantized_model_dtype}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=args.use_fast, trust_remote_code=args.trust_remote_code)
    quantize_config = BaseQuantizeConfig(bits=args.bits, group_size=args.group_size, desc_act=args.desc_act, damp_percent=args.damp_percent)
    model = AutoGPTQForCausalLM.from_pretrained(args.model_name_or_path, quantize_config=quantize_config, low_cpu_mem_usage=True, torch_dtype=torch_dtype, trust_remote_code=args.trust_remote_code)
    traindataset = get_c4(args.num_samples, tokenizer, args.seq_len)
    model.quantize(traindataset, use_triton=args.use_triton, batch_size=args.batch_size, cache_examples_on_gpu=args.cache_examples_on_gpu)
    model.save_quantized(args.output_dir, use_safetensors=True)
    tokenizer.save_pretrained(args.output_dir)

def init():
    set_seed(42)
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)

def train():
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    quantize_and_save(args)

if __name__ == "__main__":
    init()
    train()
