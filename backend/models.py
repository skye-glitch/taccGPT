from pydantic import BaseModel
from typing import List, Optional

class QA_pair(BaseModel):
    prompt: str
    answer: str
    user: str

class PromptWNumAnswers(BaseModel):
    prompt: str
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
    rankings: List[RankingResult]

class ResponseMessage(BaseModel):
    success: bool
    message: Optional[str] = None