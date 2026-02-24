from pathlib import Path
from typing import Optional

import yaml
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

    @classmethod
    def load_custom_config(cls):
        config_path = Path("config.yml")
        yaml_data = {}
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
        return cls(**yaml_data)


settings = Settings.load_custom_config()
