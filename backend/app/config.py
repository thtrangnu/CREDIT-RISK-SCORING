from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = ROOT_DIR / "ml" / "artifacts"
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT_DIR / ".env"), extra="ignore")

    database_url: str = "mysql+pymysql://app:app_password@localhost:3307/home_credit"
    model_version: str = "v1-block2-5-isotonic"
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
