"""
Install below requirements:
    !pip install -q faiss-cpu xformers einops sentence-transformers==2.2.2 transformers llama-index==0.6.0.alpha3 langchain deepspeed accelerate sentencepiece bitsandbytes torch==2.0.0 PyPDF2==3.0.1
"""


import os
# The more the CUDA visible devices, more the inference speed - as the model gets loaded on all visible devices.
os.system("export CUDA_VISIBLE_DEVICES=2")


from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain import PromptTemplate, LLMChain


model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print(f"Loading {model_name}")

if "falcon-40b-instruct" in model_name:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        trust_remote_code=True
    )
    model.config.use_cache = False
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
    )

pipeline = pipeline(
    "text-generation", # task
    model=model,
    tokenizer=tokenizer,
    # load_in_4bit=True,
    # torch_dtype=torch.bfloat16,
    max_new_tokens=200,
    do_sample=False,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})


######### Create Vector Store #########
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load data from PDF
from langchain.document_loaders import PyPDFLoader
# Take sample PDF from the Web
filepath = "https://niphm.gov.in/IPMPackages/Wheat.pdf"
loader = PyPDFLoader(filepath)
documents = loader.load_and_split()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

# Using the top open-source Embedding model from the HF leaderboard (https://huggingface.co/spaces/mteb/leaderboard)
# NOTE: We've also tried a few other embedding models: all-MiniLM-L6-v2, all-mpnet-base-v2. But e5-large-v2 found to be perform better.
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

# Find similar docs that are relevant to the question
docsearch = FAISS.from_documents(documents, embeddings)

question = "What are the dos and don'ts in IPM for Wheat?"

# Search for the similar docs
docs = docsearch.similarity_search(question, k=2)

######### Prompting to LLM for QA #########
prompt_template = """You are a helpful AI assistant. Use the context delimited by triple backticks to answer the question comprehensively. If you don't know the answer, just say that you don't know, don't try to make up an answer. If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

    Context: ```{context}```

    Question: {question}

    Answer:
"""

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
qa_chain = load_qa_chain(llm=llm, chain_type='stuff', prompt=prompt)
out_dict = qa_chain({"input_documents": docs, "question": question}, return_only_outputs=True)
print(out_dict['output_text'])
