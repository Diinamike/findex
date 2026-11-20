from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Налаштування веб-сервісу зі змінних оточення (INDEX_PATH, EMBEDDINGS_PATH)."""

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    index_path: Path | None = None
    embeddings_path: Path | None = None
    default_scorer: str = "bm25"
    log_level: str = "INFO"
    page_size: int = 10


def get_settings() -> Settings:
    return Settings()
