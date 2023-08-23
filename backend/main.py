import uvicorn


import pandas as pd


from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
import starlette.status as status
from starlette.config import Config
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import HTMLResponse
from starlette.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database import (fetch_all_qa_pairs,
                      fetch_all_rankings,
                      add_one_ranking,
                      add_one_qa_pair)
# from load_ml_model import create_chatbot

from models import (QA_pair,
                    Answers, 
                    RankingResults, 
                    PromptWNumAnswers, 
                    ResponseMessage)

app = FastAPI()

HTTP_HOST="0.0.0.0"
HTTP_PORT=9990

origins = ["http://frontend:3000",
           "http://localhost:3000",
           "http://frontend_gradio:9991",
           "http://frontend_gradio:9991/TACC_GPT"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="./static"), name="static")
templates = Jinja2Templates(directory="./templates")

@app.get('/')
def root():
    TACC_GPT_url = "http://frontend_gradio:9995/TACC_GPT"
    return RedirectResponse(url=TACC_GPT_url)

@app.post('/submit_ranking/', response_model=ResponseMessage)
async def submit_ranking(ranking_results:RankingResults):
    res = await add_one_ranking(ranking_results)
    return ResponseMessage(success=True if res else False, message="successful")

@app.post('/record_one_qa_pair/', response_model=ResponseMessage)
async def record_one_qa_pair(qa_pair:QA_pair):
    res = await add_one_qa_pair(qa_pair)
    return ResponseMessage(success=True if res else False, message="successful")

# @app.get('/submit_prompt/{prompt}', response_model=Answers)
# def submit_prompt(prompt):
#     print(prompt)
#     # return chat(message, None)
#     return Answers(answers=[f"Rank {i} result" for i in range(5)])

@app.get("/show_database_QA_collections")
async def display_database_qa_pairs(request: Request):
    # QA_pairs = QA_collection.find({"user":user_email if user_email else "Anonymous"})
    QA_pairs = await fetch_all_qa_pairs()
    QA_dict = {}
    if len(QA_pairs) > 0:
        QA_pairs_list = [QA_pair.dict() for QA_pair in QA_pairs]
        QA_df = pd.DataFrame(QA_pairs_list, columns=QA_pairs_list[0].keys())
        QA_pairs_len = len(QA_df.index)
        QA_df['index'] = pd.Series([i for i in range(QA_pairs_len)])
        QA_dict = QA_df.to_dict("records")
    
    return templates.TemplateResponse("table_template.html",{"request":request, "data":QA_dict})

@app.get("/show_database_rankings")
async def display_database_rankings(request: Request):
    # QA_pairs = QA_collection.find({"user":user_email if user_email else "Anonymous"})
    rankings = await fetch_all_rankings()
    rankings_dict = {}
    if len(rankings) > 0:
        rankings_df = pd.DataFrame(rankings, columns=rankings[0].keys())
        rankings_len = len(rankings_df.index)
        rankings_df['index'] = pd.Series([i for i in range(rankings_len)])
        rankings_dict = rankings_df.to_dict("records")
    return templates.TemplateResponse("table_template.html",{"request":request, "data":rankings_dict})


if __name__ == '__main__':
    config = uvicorn.Config(app, host=HTTP_HOST, port=HTTP_PORT, log_level="debug", reload=True)
    server = uvicorn.Server(config=config)
    server.run()