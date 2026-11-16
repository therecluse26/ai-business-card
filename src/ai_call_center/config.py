"""Settings from the environment or a .env file.

Importing this module also loads `.env` into `os.environ`, so provider SDKs that read
environment variables (Anthropic, OpenAI, Google) see the same keys as Settings.
Existing environment variables win over the file.
"""

import os
import ssl
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


def _ensure_ca_bundle() -> None:
    """Give OpenSSL a certificate bundle when the interpreter finds none.

    A uv-managed Python on NixOS has no CA file at its built-in path, so every
    TLS connection made through the standard `ssl` module (for example Pipecat's
    WebSocket to OpenAI) fails to verify. OpenSSL honours SSL_CERT_FILE, so we
    point it at certifi's bundle. A value already set in the environment wins.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    paths = ssl.get_default_verify_paths()
    if paths.cafile and os.path.exists(paths.cafile):
        return
    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()


_ensure_ca_bundle()


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

    # Telnyx (M2+). Run `bash scripts/telnyx-setup.sh` to fill these in.
    telnyx_api_key: SecretStr = SecretStr("")
    telnyx_public_key: str = ""
    telnyx_phone_number: str = ""
    telnyx_app_id: str = ""

    # Public HTTPS URL of this server (Tailscale Funnel or ngrok), no trailing slash.
    public_url: str = ""
    port: int = 8000

    # Voice legs
    openai_api_key: SecretStr = SecretStr("")
    openai_realtime_model: str = "gpt-realtime-2.1-mini"
    openai_realtime_voice: str = "marin"  # fallback when an agent spec names no voice
    # Turn detection decides how fast the model answers after the caller stops talking.
    # "server": fixed silence timer (fastest). "semantic": the model judges if the caller is done.
    openai_turn_detection: Literal["server", "semantic"] = "server"
    openai_vad_silence_ms: int = 400  # "server" only: silence before the turn ends
    openai_vad_threshold: float = 0.6  # "server" only: 0.0-1.0; higher ignores more background noise
    openai_noise_reduction: Literal["near_field", "far_field", "off"] = "near_field"  # a handset at the ear
    openai_vad_eagerness: Literal["low", "medium", "high", "auto"] = "high"  # "semantic" only

    @property
    def webhook_url(self) -> str:
        return f"{self.public_url.rstrip('/')}/webhooks/telnyx"

    @property
    def media_stream_url(self) -> str:
        base = self.public_url.rstrip("/").replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return f"{base}/media"


@lru_cache
def get_settings() -> Settings:
    return Settings()
