# Finetune
This directory contains the code to fine-tune the open-source LLM with domain specific data.
It uses the adapters like LoRA, QLoRA, PEFT to reduce the memory overhead.

## Steps to run:
- `pip install virtualenv`
- `virtualenv -p /usr/bin/python venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- `export CUDA_VISIBLE_DEVICES=<2,3,4>`  (Replace with your available GPU devices.)
- `python falcon_peft.py`