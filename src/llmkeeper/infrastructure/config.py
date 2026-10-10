from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    db_path: str = "llmkeeper.db"
    default_patience: int = 3
    default_rolling_window: int = 5
    default_alpha: float = 0.7
    default_top_k: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
