from fastapi import APIRouter

from app.rag.chains import build_demo_answer
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    answer, sources = build_demo_answer(request.question)
    return ChatResponse(answer=answer, sources=sources)
