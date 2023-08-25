from pydantic import BaseModel
from typing import List, Optional

class PromptWNumAnswers(BaseModel):
    prompt: str
    user:str
    numAnswers:int

class Prompt(BaseModel):
    prompt: str

class Answers(BaseModel):
    answers: List[str]

class RankingResult(BaseModel):
    rank: int
    rankingAnswers: List[str]

class RankingResults(BaseModel):
    prompt: str 
    user: str
    rankings: List[RankingResult]

class ResponseMessage(BaseModel):
    success: bool
    message: Optional[str] = None

class QaPair(BaseModel):
    prompt:str
    answer:str
    user:str
    date:str

class QaPairs(BaseModel):
    qaPairs:List[QaPair]

class Ranking(BaseModel):
    prompt:str
    user:str
    date:str
    answers:List[List[str]]

class Rankings(BaseModel):
    rankings:List[Ranking]