from fastapi import APIRouter

from app.rag.chains import run_fssai_rag_search
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    res = run_fssai_rag_search(request.question)
    sources = [f"{c.document_name} ({c.section})" for c in res.citations]
    return ChatResponse(answer=res.summary + "\n\n" + res.detailed_analysis, sources=sources)

