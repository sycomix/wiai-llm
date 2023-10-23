"""
Install below requirements:
    !pip install -q llama-index==0.6.0.alpha3 langchain deepspeed accelerate sentencepiece bitsandbytes torch==2.0.0 transformers gradio==3.27.0 indic-nlp-library==0.91 sentence-transformers==2.2.2 googletrans==3.1.0a0 openpyxl lxml seaborn einops xformers
"""



import os
# The more the CUDA visible devices, more the inference speed - as the model gets loaded on all visible devices.
os.system("export CUDA_VISIBLE_DEVICES=\"4,5,6\"")

from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferWindowMemory


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
    max_new_tokens=100,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})


def user(
    user_message,
    history
):
    history = history + [(user_message, None)]
    return "", history


def get_context_from_db(query):
    return f"Human: {query}"


def get_llm_chain():
    template = """As an intelligent AI assistant, give the precise answer to the question delimited by triple backticks. If you don't know the answer, just say that you do not know, don't try to make up an answer.

    Question: ```{question}```
    
    Answer:"""

    prompt = PromptTemplate(
        input_variables=["question"],
        template=template
    )

    memory = ConversationBufferWindowMemory(k=5, return_messages=True, memory_key="chat_history", ai_prefix="Chatbot")

    return LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)


def get_answer_bot(query):
    content = get_context_from_db(query)
    llm_chain = get_llm_chain()
    result = llm_chain.run(query)
    result = result.strip()

    if "do not know" in result or "don't know" in result:
        return result, ""

    return result


def bot(
    history
):
    query = history[-1][0]
    if len(query) <= 1:
        history[-1][1] = "Hi I'm an AI assistant and here to help you! Could you please elaborate more?"
    else:
        result = get_answer_bot(query)
        history[-1][1] = result 

    return history


def clear_history():
    return None


import gradio as gr

with gr.Blocks(title='Chatbot', theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column(scale=0.85):
            # chatbot = gr.Chatbot()
            chatbot = gr.Chatbot([], elem_id="chatbot").style(height=750)
            msg = gr.Textbox(
                show_label=False,
                placeholder="Type questions and press enter",
            ).style(container=False)
        
        msg.submit(
            user, [msg, chatbot], [msg, chatbot]
        ).then(bot, chatbot, chatbot)

    with gr.Row():
        clear = gr.Button("Clear")
        clear.click(
           clear_history, None, chatbot, queue=False)

demo.launch(share=True)

