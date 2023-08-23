from pydantic import BaseModel
from typing import List, Optional

class Answers(BaseModel):
    answers: List[str]

class PromptWNumAnswers(BaseModel):
    prompt: str
    numAnswers:int