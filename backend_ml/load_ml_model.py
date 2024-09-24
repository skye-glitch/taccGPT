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
from langchain.prompts import PromptTemplate
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


# This is depreated when langchain is used
def process_response(response):
    # output = str(response[0]["generated_text"])
    # output = output.split("Assistant: ")[1]
    # output = output.replace("<|endoftext|></s>", "").replace("-- <|endoftext|>", "").replace("<|endoftext|>","")
    answer = response.split("Answer: ",1)[1]
    answer = answer.split("<|endoftext|>",1)[0]
    return answer

def create_taccgpt(path, max_new_tokens):
    f = open("./log.txt","a")
    f.write(f"start creating taccgpt\n")
    f.close()
    # if os.path.exists(path):
    #     # Locally tokenizer loading has some issue, so we need to force download
    #     model_json = os.path.join(path, "config.json")
    #     if os.path.exists(model_json):
    #         model_json_file = json.load(open(model_json))
    #         model_name = model_json_file["_name_or_path"]
    #         tokenizer = AutoTokenizer.from_pretrained(model_name,
    #                                                   fast_tokenizer=True)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    #tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(path)

    f = open("./log.txt","a")
    f.write(f"get config\n")
    f.close()
    # if torch.cuda.is_available():
    #     model = OPTForCausalLM.from_pretrained(path,
    #                                        from_tf=bool(".ckpt" in path),
    #                                        config=model_config).half()
    # else:
    #     model = OPTForCausalLM.from_pretrained(path,
    #                                        from_tf=bool(".ckpt" in path),
    #                                        config=model_config)
   
    model = AutoModelForCausalLM.from_pretrained(path,
                                        from_tf=bool(".ckpt" in path),
                                        config=model_config)
    f = open("./log.txt","a")
    f.write(f"get llama model\n")
    f.close()
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    # transformers(>4.32.0) model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=4) which claims can better utilize cuda core
    # generator = pipeline("text-generation",
    #                 model=model,
    #                 tokenizer=tokenizer,
    #                 torch_dtype=torch.float16,
    #                 device='cuda' if torch.cuda.is_available() else 'cpu',
    #                 max_new_tokens=max_new_tokens)

    # f = open("./log.txt","a")
    # f.write(f"generator ready\n")
    # f.close()
    # move this to chat rag_chain.invoke("What is design safe?")


    # RAG
    # change to the whole directory after test
    loader = DirectoryLoader('./DS-User-Guide/user-guide/docs/', glob="**/*.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader)
    docs = loader.load()

    f = open("./log.txt","a")
    f.write(f"in loading doc,load {len(docs)} docs\n")
    f.close()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(docs)
    f = open("./log.txt","a")
    f.write(f"split results in len {len(texts)}\n")
    f.close()
    #change to cuda on machine with GPU resource
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={"normalize_embeddings": True},
    )
    f = open("./log.txt","a")
    f.write(f"embeddings ready\n")
    f.close()
    #query_result = embeddings.embed_query(texts[0].page_content)
    #print(f'query result len is {len(query_result)}')

    db = Chroma.from_documents(texts, embeddings, persist_directory="db")
    f = open("./log.txt","a")
    f.write(f"db ready\n")
    f.close()
    #results = db.similarity_search("design safe", k=2)
    #print(results[0].page_content)

    # From langchain tutorial
    # Retrieve and generate using the relevant snippets of the blog.
    retriever = db.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    f = open("./log.txt","a")
    f.write(f"retriever & prompt ready\n")
    f.close()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    
    async def chat(message: str, temperature: float):
        '''
        message should contain both message and user name (or other things that can distinguish users from each other)
        '''
        # processed_input = f" Human: {message}\n Assistant: " # Deepspeed chat required format
        # response = generator(processed_input, max_new_tokens=max_new_tokens)
        # processed_response = process_response(response)

        ## LangChain
        # processed_message = ''
        # for message in messages:
        #     processed_message = processed_message + f" {message.role}: {message.content}\n"
        # processed_message = processed_message + " Assisstant: "

        # response = generator({"question":message})
        # processed_response = response['text']
        # qa_pair = {"date":datetime.now().strftime("%Y %m %d"), 
        #         #    "prompt":f"{memory.load_memory_variables({})['chat_history']} Human: {message}\n Assistant: ", 
        #            "prompt":message,
        #            "user":user_email if user_email else "Anonymous", 
        #            "answer":processed_response}
        # res = requests.post(url="http://backend:9990/record_one_qa_pair/", json=qa_pair)

        ## ChatbotUI  
        f = open("./log.txt","a")
        f.write(f"in chat call \n")
        f.close()    
        # streamer = TextIteratorStreamer(tokenizer, timeout=600)
        # f = open("./log.txt","a")
        # f.write(f"streamer ready \n")
        # f.close()
        # how to send in kwargs?
        MODEL_NAME = "./nlp_models/output_1.3b_epoch32_sqLength512"
        generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
        generation_config.temperature = temperature
        generation_config.top_p = 0.95
        generation_config.top_k = 40
        generation_config.do_sample = True if temperature > 0.0 else False
        generation_config.max_new_tokens = max_new_tokens
        
        f = open("./log.txt","a")
        f.write(f"generation config ready \n")
        f.close()

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
        f = open("./log.txt","a")
        f.write(f"text_pipeline ready on gpu ?{torch.cuda.is_available()} \n")
        f.close()
        llm = HuggingFacePipeline(pipeline=text_pipeline) 
        f = open("./log.txt","a")
        f.write(f"get huggingfacepipeline ready \n")
        f.close()
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        f = open("./log.txt","a")
        f.write(f"rag_chain ready \n")
        f.close()
        # generation_kwargs = dict(text_inputs=message,
        #                         temperature=temperature, 
        #                         top_p=0.95, 
        #                         top_k=40, 
        #                         do_sample=True if temperature > 0.0 else False,
        #                         max_new_tokens=max_new_tokens,
        #                         streamer=streamer)
        # thread = Thread(target=generator, kwargs=generation_kwargs)
        # Sikan: how to use thread on this?
        #thread = Thread(target=rag_chain.invoke, args=(message,))
        #thread.start()
        #change to message after debug
        res = rag_chain.invoke(message)
        seen_start_token = False
        answer = res.split("Answer: ",1)[1]
        answer = answer.split("<|endoftext|>",1)[0]
        f = open("./log.txt","a")
        f.write(f"res {res} \n")
        f.close()
        # for token in streamer:
        #     # print(bytes(token,'utf-8'))
        #     if seen_start_token:
        #         processed_token = token.split("Assistant: ")[1] if(re.search(r"Assistant:",token)) else token
        #         processed_token = processed_token.replace("<|endoftext|></s>", "").replace("-- <|endoftext|>", "").replace("<|endoftext|>","")

        #         # Avoid sending sentence that has nothing but ' '
        #         if len(processed_token.replace(" ","")) == 0: 
        #             continue
        #         answer = answer + processed_token
        #         yield processed_token
        #         # time.sleep(0.05)

        #     # Remove the preceeding context, and only return the content following the last "Assistant:"
        #     if(re.search(r"Assistant:",token)):
        #         seen_start_token = True
        #thread.join()
        # for token in streamer:
        #     answer = answer + token
        #del streamer

        qa_pair = {"date":datetime.now().strftime("%Y %m %d"), 
                   "prompt":message,
                   "user":user_email if user_email else "Anonymous", 
                   "answer":answer}
        res = requests.post(url="http://backend:9990/record_one_qa_pair/", json=qa_pair)
        f = open("./log.txt","a")
        f.write(f"after post to database\n")
        f.close()
        return answer
    
    def generate_answers(message, numAnswers):
        # print("\n\nGenerating answers now...\n\n")
        message_repeated = [f"Human: {message}\n Assistant: " for _ in range(numAnswers)]

        # res = [process_response(x) for x in generator(message_repeated, 
        #                                               temperature=0.2, 
        #                                               top_p=0.95, 
        #                                               top_k=40, 
        #                                               do_sample=True, 
        #                                               max_new_tokens=max_new_tokens,
        #                                               batch_size=len(message_repeated))]
        f = open("./log.txt","a")
        f.write(f"in generate_answers call \n")
        f.close()    
        MODEL_NAME = "./nlp_models/output_1.3b_epoch32_sqLength512"
        generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
        generation_config.temperature = 0.2
        generation_config.top_p = 0.95
        generation_config.top_k = 40
        generation_config.do_sample = True
        generation_config.max_new_tokens = max_new_tokens
        f = open("./log.txt","a")
        f.write(f"generation config ready \n")
        f.close()
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            generation_config=generation_config,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            #device_map="auto",
        )
        f = open("./log.txt","a")
        f.write(f"text_pipeline ready on gpu {torch.cuda.is_available()}\n")
        f.close()
        llm = HuggingFacePipeline(pipeline=text_pipeline) 
        f = open("./log.txt","a")
        f.write(f"get huggingfacepipeline ready \n")
        f.close()
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        f = open("./log.txt","a")
        f.write(f"rag_chain ready \n")
        f.close()
        res = [process_response(rag_chain.invoke(x)) for x in message_repeated]
        f = open("./log.txt","a")
        f.write(f"here is the res {res} above is the res \n")
        f.close()

        return res

    return chat, generate_answers