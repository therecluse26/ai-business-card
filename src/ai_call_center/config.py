"""Settings from the environment or a .env file.

Importing this module also loads `.env` into `os.environ`, so provider SDKs that read
environment variables (Anthropic, OpenAI, Google) see the same keys as Settings.
Existing environment variables win over the file.
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_model: str = "anthropic:claude-opus-5-5"
    owner_name: str = "the owner"
    owner_phone: str = "+15555550100"
    profile_path: Path = Path("profile.md")
    leads_db: Path = Path("data/leads.sqlite3")

    # Lead emails to the owner. Never shown to an agent.
    owner_email: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: SecretStr = SecretStr("")
    smtp_from: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
