import torch


from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import TextIteratorStreamer
from transformers import BitsAndBytesConfig


# for RAG
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import DirectoryLoader
import transformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from accelerate import infer_auto_device_map
from accelerate import load_checkpoint_and_dispatch
from accelerate import init_empty_weights

# example selector for in-context learning
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from operator import itemgetter


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main(in_path="/scratch/07980/sli4/data/gpt/meta-llama/Meta-Llama-3.1-8B-Instruct/", in_MODEL_NAME="/scratch/07980/sli4/data/gpt/meta-llama/Meta-Llama-3.1-8B-Instruct/", quant = None):
    path = in_path
    tokenizer = AutoTokenizer.from_pretrained(path)   
    model_config = AutoConfig.from_pretrained(path)
    
    if quant is not None:
        if quant == "4": 
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)           
            model = AutoModelForCausalLM.from_pretrained(path,
                                    from_tf=bool(".ckpt" in path),
                                    config=model_config,
                                    device_map="auto",
                                    # quantize to fit in RAM
                                    quantization_config=quantization_config,
                                    )
        elif quant == "8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True) 
            model = AutoModelForCausalLM.from_pretrained(path,
                                    from_tf=bool(".ckpt" in path),
                                    config=model_config,
                                    device_map="auto",
                                    # quantize to fit in RAM
                                    quantization_config=quantization_config,
                                    )
        else:
            raise Exception("no matching quant format")
    else:    
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(model_config)
        device_map = infer_auto_device_map(model, max_memory={0: "30GiB", 1: "30GiB", 2: "30GiB"})
        model = load_checkpoint_and_dispatch(model, path, device_map=device_map,offload_folder="/scratch/07980/sli4/huggingface_cache/offload")
    model.eval()
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    # RAG
    loader = DirectoryLoader('./ce311k/notebooks/lectures/', glob="**/*_solutions.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader, load_hidden=False)
    # fix for ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1007)
    import ssl
    ssl._create_default_https_context = ssl._create_stdlib_context
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

    # only need to insert document once
    db = Chroma.from_documents(texts, embeddings, persist_directory="db_ce")
    #db = Chroma(persist_directory="db_ce", embedding_function=embeddings)
    #results = db.similarity_search("data structure", k=2)
    #print(results[0].page_content)

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = db.as_retriever(k=1)

    # redundant filter
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[redundant_filter]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
    )
    # retrive_res = compression_retriever.invoke(
    #     "data structure"
    # )

   

    messages = ["How to solve system of linear equations using Gauss Elimination?", "How to check length of matrix with python?", "Where can I read more about Numpy?"]
    temperature = 0.8

    MODEL_NAME = in_MODEL_NAME
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.temperature = temperature
    generation_config.top_p = 0.95
    generation_config.top_k = 40
    generation_config.do_sample = True if temperature > 0.0 else False
    generation_config.max_new_tokens = 512

    
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        generation_config=generation_config,
    )

    llm = HuggingFacePipeline(pipeline=text_pipeline) 


    for question in messages:
        if quant is not None:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a TA for students. Use the following retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use one sentences maximum and keep the answer concise. Give students hints instead of telling them the answer."),
                ("human", "{context}"),
                ("human", "{question}")
            ])
            rag_chain = (
            {"context": itemgetter("question") | compression_retriever | format_docs,
            "question": itemgetter("question")}
            | prompt
            | llm
            | StrOutputParser()
            )        
            res = rag_chain.invoke({"question":question})
            answer = res.split(question)[-1]
            answer = answer.split("<|endoftext|>")[0]
            answer = answer.replace("AI: ", '') 
            answer = answer.replace("Assistant: ", '')
            print("=====================================================")
            print("Here is the whole chain:")
            print(res)
            print("-----------------------------------------------------")
            print("Here is the answer:")
            print(answer)
            print("=====================================================")


            # remove few shot learning and RAG
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a TA for students. If you don't know the answer, just say that you don't know. Use one sentences maximum and keep the answer concise. Give students hints instead of telling them the answer."),
                ("human", "{question}")
            ])
            chain = prompt.pipe(llm)       
            res = chain.invoke({"question":question})
            answer = res.split(question)[-1]
            answer = answer.split("<|endoftext|>")[0]
            answer = answer.replace("AI: ", '') 
            answer = answer.replace("Assistant: ", '') 
            print("=====================================================")
            print("Remove RAG. Here is the whole chain:")
            print(res)
            print("-----------------------------------------------------")
            print("Here is the answer:")
            print(answer)
            print("=====================================================")

        else:
            results = db.similarity_search(question, k=2)
            retrieved_context = format_docs(results)
            prompt = f"system: You are a TA for students. Use the following retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use one sentences maximum and keep the answer concise. Give students hints instead of telling them the answer.\"\n human:{retrieved_context} \n human: {question}"
            inputs = tokenizer(prompt,return_tensors="pt").to(0)
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True)[0]
            print("Here is the answer:")
            print(tokenizer.decode(outputs.cpu().squeeze()).split(question)[-1])
            print("=====================================================")
            

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Put in model name and path')
    parser.add_argument('--path', metavar='path', 
                        help='the path to model')
    parser.add_argument('--MODEL_NAME', metavar='path', 
                        help='name of model')
    parser.add_argument('--quant',  help='quantization scheme')
    args = parser.parse_args()
    main(in_path=args.path, in_MODEL_NAME=args.MODEL_NAME, quant=args.quant)