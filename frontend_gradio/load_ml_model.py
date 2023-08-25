import torch
import os
import json
import time
import sys
import requests
from datetime import datetime
from transformers import pipeline
from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer

# TODO: change direct call to fastapi call
# sys.path.append("../backend_db")
# from backend_db.database import add_one_qa_pair

user_email = None

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
                         torch_dtype=torch.float16)
    
    async def chat(message, history):
        processed_input = f"Human: {message}\n Assistant: " # Deepspeed chat required format
        response = generator(processed_input, do_sample=True, max_new_tokens=max_new_tokens)
        processed_response = process_response(response)
        
        res = requests.post(url="http://backend:9990/record_one_qa_pair/", data={"date":datetime.now().strftime("%Y %m %d"), 
                                                                                 "prompt":message, 
                                                                                 "user": user_email if user_email else "Anonymous", 
                                                                                 "answer":processed_response})

        ## streaming reply effect (implemented below in the Blocks())
        for i in range(len(processed_response)):
            time.sleep(0.02)
            yield processed_response[:i+1]
        # return processed_response
    
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
                                           config=model_config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    # transformers(>4.32.0) model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=4) which claims can better utilize cuda core
    generator = pipeline("text-generation",
                         model=model,
                         tokenizer=tokenizer,
                         torch_dtype=torch.float16)
    
    def generate_answers(message, numAnswers):
        message_repeated = [f"Human: {message}\n Assistant: " for _ in range(numAnswers)]

        res = [process_response(x) for x in generator(message_repeated, temperature=0.2, top_p=0.95, top_k=50, do_sample=True, max_new_tokens=max_new_tokens)]
        return res

    return generate_answers