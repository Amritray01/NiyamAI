from functools import lru_cache
from os import getenv


class Settings:
    def __init__(self) -> None:
        self.fastapi_host = getenv("FASTAPI_HOST", "0.0.0.0")
        self.fastapi_port = int(getenv("FASTAPI_PORT", "8000"))
        self.cors_origins = [
            origin.strip()
            for origin in getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
            if origin.strip()
        ]
        # NeonDB PostgreSQL Connection
        self.database_url = getenv(
            "DATABASE_URL",
            "postgresql://user:password@ep-sample-123456.us-east-2.aws.neon.tech/neondb?sslmode=require"
        )
        
        # Qdrant Cloud Connection
        self.qdrant_url = getenv("QDRANT_URL", "https://your-cluster-id.us-east4-0.gcp.cloud.qdrant.io:6333")
        self.qdrant_api_key = getenv("QDRANT_API_KEY", "")
        self.qdrant_collection = getenv("QDRANT_COLLECTION", "fssai_regulations")
        
        # LLM Provider Keys
        self.google_api_key = getenv("GOOGLE_API_KEY", "")
        self.openai_api_key = getenv("OPENAI_API_KEY", "")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
