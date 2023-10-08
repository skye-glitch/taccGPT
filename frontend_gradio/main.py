import gradio as gr
import uvicorn
from fastapi import FastAPI
import argparse
from fastapi.middleware.cors import CORSMiddleware

from load_ml_model import create_taccgpt_rank, create_taccgpt_chat
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
    processed_message = ''
    for message in chatMessage.messages:
        processed_message = processed_message + f" {message.role}: {message.content}\n"

    processed_message = processed_message + " Assistant: "

    threadLimiter.acquire()
    try:
        content=chatbot(processed_message, chatMessage.temperature)
    finally:
        threadLimiter.release()
    return EventSourceResponse(sep="\r\n",
                               status_code=200,
                               content=content,
                            #    content=chatbot(processed_message, chatMessage.temperature),
                               headers={'Content-Type': 'text/plain',})
    #ping=600  



@app.post('/submit_prompt/',response_model=Answers)
def submit_propmt(message:PromptWNumAnswers):
    global generate_multiple_answers
    res = generate_multiple_answers(message.prompt, message.numAnswers)
    return Answers(answers=res)

def load_taccgpt_rank(args):
    global generate_multiple_answers
    generate_multiple_answers = create_taccgpt_rank(args.path, args.max_new_tokens)

def load_taccgpt_chatbot(args):
    global chatbot
    chatbot = create_taccgpt_chat(args.path, args.max_new_tokens)

# def mount_gradio_ChatInterface(args):
#     global chatbot
#     chatbot = create_taccgpt_chat(args.path, args.max_new_tokens)

#     tacc_gpt_interface = gr.ChatInterface(fn=chatbot,
#                       chatbot=gr.Chatbot(height=700,show_copy_button=True),
#                       textbox=gr.Textbox(placeholder="Welcome to use TACC GPT, please type your question here.", 
#                                          container=False, scale=20),
#                       title="TACC GPT",
#                       theme="soft")
#     tacc_gpt_interface = tacc_gpt_interface.queue()
#     chatbot_app = gr.mount_gradio_app(app, tacc_gpt_interface, path=CUSTOM_PATH, gradio_api_url=f"http://{args.http_host}:{args.http_port}{CUSTOM_PATH}/")

#     # config = uvicorn.Config(chatbot_app, host=args.http_host, port=args.http_port, log_level="debug", reload=True)
#     config = uvicorn.Config(app, host=args.http_host, port=args.http_port, reload=True)
#     server = uvicorn.Server(config=config)
#     return server

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

    load_taccgpt_rank(args)
    load_taccgpt_chatbot(args)

    # server = mount_gradio_ChatInterface(args)
    config = uvicorn.Config(app, host=args.http_host, port=args.http_port, reload=True)
    server = uvicorn.Server(config=config)
    server.run()
