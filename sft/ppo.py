import os
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    pipeline, 
    AutoModelForSequenceClassification, 
    AutoTokenizer
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

tqdm.pandas()


@dataclass
class ScriptArguments:

    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for input"})
    output_min_length: Optional[int] = field(default=32, metadata={"help": "minimum length for generation"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="./checkpoints/tuning_llama_rl/",
                                      metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    lora_r: Optional[int] = field(default=16, metadata={"help": 'Lora attention dimension (the "rank")'})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "The alpha parameter for Lora scaling."})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "The dropout probability for Lora layers."})
    top_k: Optional[float] = field(default=0.0, metadata={"help": "Topk for generation."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Topp for generation."})


def build_dataset(script_args, tokenizer, dataset_name, input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    train_dataset = load_dataset(dataset_name, split="train")
    original_columns = train_dataset.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["text"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < script_args.max_length, batched=False)

    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def train():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    config = PPOConfig(
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=script_args.early_stopping,
        target_kl=script_args.target_kl,
        ppo_epochs=script_args.ppo_epochs,
        seed=script_args.seed,
    )

    if "decapoda" in script_args.model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": DEFAULT_PAD_TOKEN,
            }
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = DEFAULT_PAD_TOKEN
            tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = build_dataset(script_args, tokenizer, script_args.dataset_name)

    current_device = Accelerator().local_process_index

    lora_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        load_in_8bit=True,
        device_map={"": current_device},
        peft_config=lora_config,
    )

    optimizer = None
    if script_args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )

    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"

    generation_kwargs = {
        "top_k": script_args.top_k,
        "top_p": script_args.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    output_min_length = script_args.output_min_length
    output_max_length = script_args.output_max_length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)


    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        question_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        reward_name = script_args.reward_model_name
        rank_model, rank_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
        rank_model.to(device)
        rewards = []
        for q, r in zip(batch["query"], batch["response"]):
            inputs = rank_tokenizer(q, r, return_tensors='pt').to(device)
            score = rank_model(**inputs).logits[0].cpu().detach() - script_args.reward_baseline
            rewards.append(score)
        
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")    


if __name__ == "__main__":
    train()
