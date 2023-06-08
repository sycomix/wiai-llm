"""
Install below requirements:
    pip install llama-index==0.6.0.alpha3
    pip install langchain
    pip install deepspeed
    pip install accelerate
    pip install sentencepiece
    pip install bitsandbytes
    pip install torch==2.0.0
    pip install transformers
    pip install gradio==3.27.0
    pip install indic-nlp-library==0.91
    pip install sentence-transformers==2.2.2
    pip install googletrans==3.1.0a0
    pip install openpyxl
    pip install lxml
    pip install seaborn
    pip install einops
    pip install xformers
"""


from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferWindowMemory


model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

pipeline = pipeline(
    "text-generation", # task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    # load_in_4bit=True,
    device_map="auto",
    max_length=200,
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
    history = history + [("Question: " + user_message, None)]
    return "", history


def get_context_from_db(query):
    return f"Human: {query}"


def get_llm_chain():
    template = """
    You are an intelligent chatbot. Help the following question with brilliant answers.
    Question: {question}
    Answer:"""

    prompt = PromptTemplate(
        input_variables=["question"],
        template=template
    )

    memory = ConversationBufferWindowMemory(k=3, return_messages=True, memory_key="chat_history", ai_prefix = "Chatbot")

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False
    )

    return llm_chain


def get_answer_bot(query):
    content = get_context_from_db(query)
    llm_chain = get_llm_chain()
    result = llm_chain.run(query)
    result = result.strip()

    if "do not know" in result:
        return result, ""

    return result


def bot(
    history
):
    query = history[-1][0]
    if len(query) <= 1:
        history[-1][1] = "Hi I'm AI assistant! Here to help you. Could you please elaborate more?"
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

