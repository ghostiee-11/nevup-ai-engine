from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://nevup:nevup@localhost:5432/nevup"
    jwt_secret: str = "97791d4db2aa5f689c3cc39356ce35762f0a73aa70923039d8ef72a2840a1b02"
    gemini_api_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_embed_model: str = "models/text-embedding-004"
    gemini_profile_model: str = "models/gemini-1.5-flash"
    embedding_dim: int = 768
    seed_path: str = "/data/nevup_seed_dataset.json"
    log_level: str = "INFO"


settings = Settings()
