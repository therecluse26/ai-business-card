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
    summary_model: str = "anthropic:claude-haiku-5-5"  # writes the two-sentence call summary at hangup
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

    # Guardrails for a public number. 0 turns a limit off.
    max_call_minutes: int = 10  # the agent warns one minute before; the call ends at the limit
    max_calls_per_number_per_day: int = 5  # the owner's phone (OWNER_PHONE) is exempt
    max_concurrent_calls: int = 3
    daily_minutes_cap: int = 120  # all callers together, per 24 hours; this bounds the daily model cost
    transfer_hours: str = "09:00-18:00"  # US Eastern; empty = a live transfer is allowed at any time
    transfer_days: str = "mon-fri"
    transfer_ring_secs: int = 25  # how long the owner's phone rings before the agent comes back to the caller

    # Text messages (off until the number has a Telnyx messaging profile and a 10DLC campaign).
    sms_enabled: bool = False
    portfolio_url: str = ""  # the link an agent can text to a caller

    # Number lookup on each new caller (Telnyx bills each lookup; results are cached per number).
    caller_lookup: Literal["off", "carrier", "carrier+name"] = "carrier+name"

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

    # One loudness for every agent voice, in dBFS (speech level). Voices differ by up to 10 dB,
    # most of all across providers. 0 turns the leveler off.
    voice_level_dbfs: float = -22.0

    # Gemini Live voice leg. Without GOOGLE_API_KEY, agents set to gemini_live run on OpenAI.
    google_api_key: SecretStr = SecretStr("")
    gemini_live_model: str = "models/gemini-3.8-live"
    gemini_vad_silence_ms: int = 400  # silence before the turn ends
    gemini_vad_start_sensitivity: Literal["low", "high"] = "low"  # "low" ignores more background noise

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
