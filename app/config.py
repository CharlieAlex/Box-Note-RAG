from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mlflow_tracking_uri: str = "sqlite:///data/mlflow.db"
    mlflow_experiment_name: str = "My_Notes_RAG_Agent"
    enable_telemetry: bool = True
    ollama_model: str
    embeddings_model: str
    chroma_path: str
    batch_size: int
    openai_api_key: Optional[str] = None
    debug: bool = False

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
