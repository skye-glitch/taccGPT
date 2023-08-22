import gradio as gr
import uvicorn
from fastapi import FastAPI
import argparse

from load_ml_model import create_chatbot
from fastapi.responses import RedirectResponse

CUSTOM_PATH = "/TACC_GPT"
app = FastAPI()

@app.get('/')
def root():
    return RedirectResponse(url=CUSTOM_PATH)

def mount_gradio_ChatInterface(args):
    global chat
    chat = create_chatbot(args.path, args.max_new_tokens)
    tacc_gpt_interface = gr.ChatInterface(fn=chat,
                      chatbot=gr.Chatbot(height=800,show_copy_button=True),
                      textbox=gr.Textbox(placeholder="Welcome to use TACC GPT, please type your question here.", 
                                         container=False, scale=15),
                      title="TACC GPT",
                      theme="soft")
    tacc_gpt_interface = tacc_gpt_interface.queue()
    chatbot_app = gr.mount_gradio_app(app, tacc_gpt_interface, path=CUSTOM_PATH, gradio_api_url=f"http://{args.http_host}:{args.http_port}{CUSTOM_PATH}/")

    config = uvicorn.Config(chatbot_app, host=args.http_host, port=args.http_port, log_level="debug", reload=True)
    server = uvicorn.Server(config=config)
    return server

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
    parser.add_argument('--http_port', type=int, default=9991)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    server = mount_gradio_ChatInterface(args)
    server.run()
