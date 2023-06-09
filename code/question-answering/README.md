# Question-Answering
This directory contains the code to build the Question-Answering pipeline using the custom (open-source) LLM with domain specific data.

## Steps to run:
- `pip install virtualenv`
- `virtualenv -p /usr/bin/python venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- `export CUDA_VISIBLE_DEVICES=<2,3,4>`  (Replace with your available GPU devices.)
- `python Retrieval_QA_Falcon7B.py`