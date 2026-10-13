from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    test_database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag_test"
    default_daily_quota: int = 50


settings = Settings()
