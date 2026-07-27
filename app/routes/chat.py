from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.rag import ask_question,chat_with_function_calling
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

class Question(BaseModel):
    question: str
    n_results: int = 3  
    
class ChatResponse(BaseModel):
    answer: str
    used_rag: bool
    conversation_history: list

class ChatMessage(BaseModel):
    message: str
    conversation_history: Optional[list] = None
    temprature: float=0.7

@router.post("/ask")
async def ask(q: Question):
    try:
        result = await ask_question(q.question, q.n_results)
        return result
    
    except Exception:
        logger.exception("ask failed")
        raise HTTPException(status_code=500, detail="Internal server error")
    

@router.post("/chat", response_model=ChatResponse)
async def chat(msg: ChatMessage):
    try:
        result = await chat_with_function_calling(msg.message,msg.conversation_history,msg.temprature)
        return result
    
    except Exception:
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail="Internal server error")
