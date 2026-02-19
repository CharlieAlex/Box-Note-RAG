from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_model: str
    embeddings_model: str
    chroma_path: str
    batch_size: int
    openai_api_key: Optional[str] = None
    debug: bool = False

    class Config:
        env_file = ".env"


settings = Settings()
