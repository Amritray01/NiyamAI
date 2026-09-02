from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.chat import router as chat_router
from app.api.routes.search import router as search_router
from app.api.routes.news import router as news_router
from app.core.config import settings

app = FastAPI(
    title="NiyamAI - FSSAI Regulatory Search Platform API",
    description="Intelligent search, citations, and live regulatory updates for FSSAI compliance.",
    version="0.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(news_router, prefix="/api")
