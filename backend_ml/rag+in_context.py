import torch
import os
import json
import time
import sys
import requests
from datetime import datetime
from models import Message
from typing import List, Optional
import re

from LLMChainProcessOutput import LLMChainProcessOutput
from langchain.memory import RedisChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler

from transformers import pipeline
from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer, LlamaForCausalLM,AutoModelForCausalLM
from transformers import TextIteratorStreamer

from concurrent.futures import ThreadPoolExecutor
from threading import Thread
# Will be added after AUTH is finished
user_email = None

# for RAG
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import DirectoryLoader
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from textwrap import fill

# example selector for in-context learning
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from operator import itemgetter


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    path = "/etc/TACC_GPT/nlp_models/Meta-Llama-3.1-8B-Instruct/"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model_config = AutoConfig.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path,
                                    from_tf=bool(".ckpt" in path),
                                    config=model_config)
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    # RAG
    # todo: use all content in the directory
    # loader = DirectoryLoader('./DS-User-Guide/user-guide/docs/', glob="**/*.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader)
    loader = DirectoryLoader('./DS-User-Guide/user-guide/docs/analysis', glob="**/*.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(docs)
    #change to cuda on machine with GPU resource
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={"normalize_embeddings": True},
    )
    #query_result = embeddings.embed_query(texts[0].page_content)
    #print(f'query result len is {len(query_result)}')

    db = Chroma.from_documents(texts, embeddings, persist_directory="db")
    #results = db.similarity_search("design safe", k=2)
    #print(results[0].page_content)

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = db.as_retriever()

    examples = [
        {"input": "2_happy", "output": "4_sad"},
        {"input": "6_tall", "output": "8_short"},
        {"input": "10_energetic", "output": "12_lethargic"},
        {"input": "14_sunny", "output": "16_gloomy"},
        {"input": "18_windy", "output": "20_calm"},
    ]

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        # This is the list of examples available to select from.
        examples,
        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
        embeddings,
        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
        Chroma,
        # This is the number of examples to produce.
        k=1,
    )

    message = "What is design safe project?"
    temperature = 0.2

    MODEL_NAME = "./nlp_models/output_1.3b_epoch32_sqLength512"
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.temperature = temperature
    generation_config.top_p = 0.95
    generation_config.top_k = 40
    generation_config.do_sample = True if temperature > 0.0 else False
    generation_config.max_new_tokens = 128

    # streamer added by Sikan
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        generation_config=generation_config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        #device_map="auto",
        # streamer=streamer,
    )

    llm = HuggingFacePipeline(pipeline=text_pipeline) 
    question = message
    selected_examples = example_selector.select_examples({"question": question})
    print(f"Examples most similar to the input: {question}")
    for example in selected_examples:
        print("\n")
        for k, v in example.items():
            print(f"{k}: {v}")

    # rag_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
    few_shot_template = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("assistant", "{output}")
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=few_shot_template,
        examples=selected_examples,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of examples and retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise"),
        few_shot_prompt,
        ("human", "{question}"),
        ("human", "{context}")
    ])
    shot_rag_chain = (
    {"context": itemgetter("question") | retriever | format_docs,
    "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
    )        
    
    print(shot_rag_chain.invoke({"question":message}))
    # res = shot_rag_chain.invoke({"question":message})
    # answer = res.split("Assistant: ",1)[1]
    # print(res)

if __name__=="__main__":
    main()