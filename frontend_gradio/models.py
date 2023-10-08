from pydantic import BaseModel
from typing import List, Optional

class Answer(BaseModel):
    answer:str

class Answers(BaseModel):
    answers: List[str]

class PromptWNumAnswers(BaseModel):
    prompt: str
    numAnswers:int


class Message(BaseModel):
    role: str
    content: str

class ChatBody(BaseModel):
    messages: List[Message]
    # message: Message
    temperature: float
    prompt: str

