########################################################################
# This is a fully working simple example to use trl's RewardTrainer.
#
# This example fine-tunes any causal language model (GPT-2, GPT-Neo, etc.)
# by using the RewardTrainer from trl, we will leverage PEFT library to finetune
# adapters on the model.
#
# Reference taken from: https://gist.github.com/pacman100/1731b41f7a90a87b457e8c5415ff1c14
########################################################################

import os
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from peft.tuners.lora import LoraLayer

from trl import SFTTrainer


def print_gpu_memory_summary():
    print(torch.cuda.memory_summary(device=None, abbreviated=False))


def clear_gpu_memory():
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print_gpu_memory_summary()


def get_gpu_device_info():
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")


clear_gpu_memory()
get_gpu_device_info()

#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ["WANDB__SERVICE_WAIT"] = "300"


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})

    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=512)
    
    model_name: Optional[str] = field(
        default="tiiuae/falcon-7b",
        metadata={
            "help": "The model that you want to train from the HuggingFace hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )

    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )

    data_files: Optional[str] = field(
        default="openassistant_best_replies_train.jsonl",
        metadata={"help": "Data files to load and model to be fine-tuned on."},
    )

    output_dir: Optional[str] = field(
        default="./checkpoints",
        metadata={"help": "Output directory path where model checkpoints would be saved."},
    )

    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )

    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )

    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )

    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )

    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )

    max_steps: int = field(default=1000, metadata={"help": "How many optimizer update steps to take"})

    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )

    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )

    packing: Optional[bool] = field(
        default=True,
        metadata={"help": "Use packing dataset creating."},
    )

    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )

    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )

    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )

    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})

    group_by_length: bool = field(
        default=False,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )

    save_steps: int = field(default=100, metadata={"help": "Save checkpoint every X updates steps."})
    
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates steps."})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )

    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],  # , "word_embeddings", "lm_head"],
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


output_dir = f"{script_args.output_dir}_{script_args.model_name.split('/')[-1]}_{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')}"
print(f"Model checkpoints would be saved at: {output_dir}")

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    num_train_epochs=script_args.num_train_epochs,
    warmup_ratio=script_args.warmup_ratio,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
)

print("Training arguments:")
print("="*100)
print(training_arguments)
print("="*100)

model, peft_config, tokenizer = create_and_prepare_model(script_args)
model.config.use_cache = False
# Print GPU memory summary
print_gpu_memory_summary()

# data_files = {"train": os.path.join(script_args.dataset_name, script_args.data_files)}
# print(f"data_files to load: {data_files}")
# dataset = load_dataset("json", data_files=data_files, split="train")
dataset = load_dataset(script_args.dataset_name, split="train")


# Reference from: https://huggingface.co/docs/trl/main/en/sft_trainer#customize-your-prompts-using-packed-dataset
def formatting_func(example):
    return f"### Instruction: {example['instruction']}\n ### Input: {example['input']}\n ### Output: {example['output']}"

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_func,
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)


for name, module in trainer.model.named_modules():
    if isinstance(module, LoraLayer):
        if script_args.bf16:
            module = module.to(torch.bfloat16)
    if "norm" in name:
        module = module.to(torch.float32)
    if "lm_head" in name or "embed_tokens" in name:
        if hasattr(module, "weight"):
            if script_args.bf16 and module.weight.dtype == torch.float32:
                module = module.to(torch.bfloat16)

trainer.train()
