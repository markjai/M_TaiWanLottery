"""Application configuration using Pydantic Settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/taiwan_lottery"
    DATABASE_URL_SYNC: str = "postgresql+psycopg2://postgres:password@localhost:5432/taiwan_lottery"

    # App
    APP_NAME: str = "Taiwan Lottery Predictor"
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "change-me"

    # ML
    MODEL_ARTIFACTS_DIR: Path = Path("./model_artifacts")
    TORCH_DEVICE: str = "cpu"

    # Scraper
    SCRAPER_ENABLED: bool = True
    BINGO_SCRAPE_URL: str = "https://www.taiwanlottery.com/lotto/bingobingo/history.aspx"


settings = Settings()
