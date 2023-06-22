# Reference taken from: https://huggingface.co/smangrul/falcon-40B-int4-peft-lora-sfttrainer

from dataclasses import dataclass, field
from typing import Optional

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

from trl import SFTTrainer

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=False,
)

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b", quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
# Model checkpoint path. Replace "checkpoint-250" with the latest checkpoint-number directory
model_checkpoints = "~/falcon_peft_checkpoints/checkpoint-250"
model_id = model_checkpoints
model = PeftModel.from_pretrained(model, model_id)

query = '### Instruction: Tell me about the latest diseases.\n### Input: Children diseases.\n### Output:'

outputs = model.generate(input_ids=tokenizer(query, return_tensors="pt").input_ids,  max_new_tokens=256,  temperature=0.5,  top_p=0.9, do_sample=True)

print(tokenizer.batch_decode(outputs))
