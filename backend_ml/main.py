# import gradio as gr
import uvicorn
from fastapi import FastAPI
import argparse
from fastapi.middleware.cors import CORSMiddleware

# from load_ml_model import create_taccgpt_rank, create_taccgpt_chat
from load_ml_model import create_taccgpt
from fastapi.responses import RedirectResponse
from models import Answers, PromptWNumAnswers, Answer, ChatBody
from sse_starlette import EventSourceResponse
import time
from fastapi.responses import StreamingResponse

from threading import Thread, BoundedSemaphore

# CUSTOM_PATH = "/TACC_GPT_UI"
app = FastAPI()


origins = ["http://frontend:3000",
           "http://localhost:3000",
           "http://localhost/Ranking"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None
generate_multiple_answers = None
NUM_THREADS_ALLOWED = 2
threadLimiter = BoundedSemaphore(NUM_THREADS_ALLOWED) # limit the number of running threads

# @app.get('/')
# def root():
#     return RedirectResponse(url=CUSTOM_PATH)

@app.post('/chatbot/')
async def chat(chatMessage:ChatBody):
    f = open("./log.txt", "a")
    f.write("in chat function\n")
    f.close()
    processed_message = ''
    for message in chatMessage.messages:
        processed_message = processed_message + f" {message.role}: {message.content}\n"

    processed_message = processed_message + " Assistant: "

    threadLimiter.acquire()
    try:
        content= await chatbot(processed_message, chatMessage.temperature)
    finally:
        threadLimiter.release()

    f = open("log.txt","a")
    f.write("chat result\n")
    f.write("content")
    f.write(str(content))
    f.write(f"{processed_message} {chatMessage.temperature}")
    f.write("\n")
    f.close()
    return EventSourceResponse(sep="\r\n",
                               status_code=200,
                               content=content,
                               headers={'Content-Type': 'text/plain',})
    #ping=600  



@app.post('/submit_prompt/',response_model=Answers)
def submit_propmt(message:PromptWNumAnswers):
    f = open("./log.txt","a")
    f.write("in submit prompt\n")
    f.close()
    global generate_multiple_answers
    res = generate_multiple_answers(message.prompt, message.numAnswers)
    f = open("./log.txt","a")
    f.write("generate multiple answers\n")
    f.write("answers are")
    f.write(str(res))
    f.write("\n")
    f.close()
    return Answers(answers=res)



def load_taccgpt(args):
    global chatbot, generate_multiple_answers
    chatbot, generate_multiple_answers = create_taccgpt(args.path,args.max_new_tokens)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        type=str,
                        help="Directory containing trained actor model")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per response",
    )
    parser.add_argument('--http_host', default='0.0.0.0')
    parser.add_argument('--http_port', type=int, default=9990)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # No enough memory to load two models into 3080 12GB, if model is opt-2.7B
    # load_taccgpt_rank(args)
    # load_taccgpt_chatbot(args)

    open('./log.txt', 'w').close()
    f = open("./log.txt","a+")
    f.write("in main\n")
    f.close()
    load_taccgpt(args)
    f = open("./log.txt","a+")
    f.write("load model\n")
    f.close()
    # server = mount_gradio_ChatInterface(args)
    config = uvicorn.Config(app, host=args.http_host, port=args.http_port, reload=True)
    server = uvicorn.Server(config=config)
    f = open("./log.txt","a")
    f.write("before run server\n")
    f.close()
    server.run()
