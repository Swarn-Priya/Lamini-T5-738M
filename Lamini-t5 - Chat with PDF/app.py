import streamlit as st
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetreivalQA
from constants import CHROMA_SETTINGS

checkpoint = 'LaMini-T5-738M'
tokenizer = AutoTokenizer.from_pretrained(ck=checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype = torch.float32
)

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer = tokenizer,
        max_length =256,
        do_sample =True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

def qa_llm():
    llm=llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name='LaMini-T5-738M')
    db = Chroma(persist_directory= "db",embedding_function=embeddings,client_settings=CHROMA_SETTINGS)
    retriver=db.as_retriever()
    qa = RetreivalQA.from_chaintype(
        llm=llm,
        chain_type="stuff",
        retriver=retriver,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']

