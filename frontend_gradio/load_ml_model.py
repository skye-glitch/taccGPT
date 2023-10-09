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

from langchain.llms import HuggingFacePipeline
from LLMChainProcessOutput import LLMChainProcessOutput
from langchain.memory import RedisChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

from transformers import pipeline
from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer
from transformers import TextStreamer, TextIteratorStreamer

from concurrent.futures import ThreadPoolExecutor
from threading import Thread
# Will be added after AUTH is finished
user_email = None



# This is depreated when langchain is used
def process_response(response):
    output = str(response[0]["generated_text"])
    output = output.split("Assistant: ")[1]
    output = output.replace("<|endoftext|></s>", "").replace("-- <|endoftext|>", "").replace("<|endoftext|>","")
    return output

def create_taccgpt_chat(path, max_new_tokens):
    if os.path.exists(path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file["_name_or_path"]
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                      fast_tokenizer=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)

    tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(path)
    if torch.cuda.is_available():
        model = OPTForCausalLM.from_pretrained(path,
                                           from_tf=bool(".ckpt" in path),
                                           config=model_config).half()
    else:
        model = OPTForCausalLM.from_pretrained(path,
                                           from_tf=bool(".ckpt" in path),
                                           config=model_config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    # transformers(>4.32.0) model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=4) which claims can better utilize cuda core
    generator = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float16,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    max_new_tokens=max_new_tokens)
    
    # # LangChain
    # pipe = HuggingFacePipeline(pipeline=pipe)
    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    ## history = RedisChatMessageHistory(url='redis://redis:6379/0',session_id='', key_prefix='summary_buffer')
    ## history.clear()
    ## memory = ConversationBufferWindowMemory(chat_memory=history, k = 5, memory_key='chat_history', ai_prefix="Assistant") # keep the last 5 interactions as context

    # prompt_template = '''{question}'''
    # prompt=PromptTemplate.from_template(prompt_template)
    # generator=LLMChainProcessOutput(
    #     llm=pipe,
    #     prompt=prompt,
    #     callback_manager=callback_manager,
    #     # memory=memory,
    #     llm_kwargs={"max_new_tokens":max_new_tokens}
    # )
    pool = ThreadPoolExecutor(max_workers=1)
    
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
        streamer = TextIteratorStreamer(tokenizer, timeout=60)
        generation_kwargs = dict(text_inputs=message,
                                temperature=temperature, 
                                top_p=0.95, 
                                top_k=40, 
                                do_sample=True if temperature > 0.0 else False,
                                max_new_tokens=max_new_tokens,
                                streamer=streamer)
        thread = Thread(target=generator, kwargs=generation_kwargs)
        thread.start()
        seen_start_token = False
        for token in streamer:
            # print(bytes(token,'utf-8'))
            if seen_start_token:
                processed_token = token.split("Assistant: ")[1] if(re.search(r"Assistant:",token)) else token
                processed_token = processed_token.replace("<|endoftext|></s>", "").replace("-- <|endoftext|>", "").replace("<|endoftext|>","")

                # Avoid sending sentence that has nothing but ' '
                if len(processed_token.replace(" ","")) == 0: 
                    continue
                yield processed_token
                # time.sleep(0.05)

            # Remove the preceeding context, and only return the content following the last "Assistant:"
            if(re.search(r"Assistant:",token)):
                seen_start_token = True
        # thread.join()
        del streamer

    return chat

def create_taccgpt_rank(path, max_new_tokens):
    if os.path.exists(path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file["_name_or_path"]
            tokenizer = AutoTokenizer.from_pretrained(model_name,fast_tokenizer=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
    
    tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(path)
    model = OPTForCausalLM.from_pretrained(path,
                                           from_tf=bool(".ckpt" in path),
                                           config=model_config).half()

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    # transformers(>4.32.0) model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=4) which claims can better utilize cuda core
    generator = pipeline("text-generation",
                         model=model,
                         tokenizer=tokenizer,
                         torch_dtype=torch.float16,
                         device='cuda')
    
    def generate_answers(message, numAnswers):
        print("\n\nGenerating answers now...\n\n")
        message_repeated = [f"Human: {message}\n Assistant: " for _ in range(numAnswers)]

        res = [process_response(x) for x in generator(message_repeated, 
                                                      temperature=0.2, 
                                                      top_p=0.95, 
                                                      top_k=40, 
                                                      do_sample=True, 
                                                      max_new_tokens=max_new_tokens,
                                                      batch_size=len(message_repeated))]
        return res

    return generate_answers