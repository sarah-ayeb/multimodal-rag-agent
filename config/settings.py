from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "dev"
    database_url: str = "postgresql://assistant:assistant@localhost:5432/assistant"
    qdrant_url: str = "http://localhost:6333"
    upload_dir: str = "data/uploads"
    max_file_size_mb: int = 500
    allowed_extensions: str = "pdf,txt,mp4,mkv,avi,mov,mp3,wav"

    # Modeles : choisis phase par phase
    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    embedding_model: str = ""
    reranker_model: str = ""
    whisper_model: str = ""

    @property
    def extensions(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_extensions.split(",") if e.strip()}


settings = Settings()
