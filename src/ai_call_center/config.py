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
    owner_name_pronunciation: str = ""  # optional; how the agents say the owner's name, in plain words
    brand_name: str = ""  # optional; the greeting then says "<brand>, <owner>'s AI business card"
    owner_phone: str = "+15555550100"
    profile_path: Path = Path("profile.md")
    leads_db: Path = Path("data/leads.sqlite3")
    # Save the caller's side of each call as a WAV file, for tests of noise and coughs. Off by default:
    # it puts callers' voices on this disk.
    save_caller_audio: bool = False
    caller_audio_dir: Path = Path("data/caller_audio")
    # Everything the server prints also goes to this file, so a call can be studied after it ended.
    # At start the file of the run before is kept as <name>.previous. Empty = no file.
    server_log: str = "data/server.log"

    # Lead emails to the owner. Never shown to an agent.
    owner_email: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: SecretStr = SecretStr("")
    smtp_from: str = ""

    # LangSmith tracing (see tracing.py). The langsmith client reads LANGSMITH_PROJECT from the environment.
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr = SecretStr("")

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
    # The owner's voicemail answers the call like a person does. With this on, Telnyx listens to
    # the first seconds after the answer; when it hears a machine, the server hangs up the
    # owner's leg and the contact agent comes back to the caller.
    transfer_voicemail_detection: bool = True
    # Call screening. The server calls the owner as a separate call; the caller hears a ring tone
    # and never hears the owner's phone. A voice tells the owner who is calling and asks for a
    # key: 1 connects the two calls, 2 or no key sends the caller back to the contact agent.
    # A voicemail cannot press 1. False = the plain transfer, with the voicemail detection above.
    transfer_screening: bool = True
    transfer_accept_secs: int = 6  # how long the owner has to press a key after the question; it is asked twice
    # What the caller hears while they wait: a WAV or MP3 file. Empty = a ring tone made by this server.
    transfer_hold_audio_url: str = ""
    # When no voice provider answers, a built-in voice apologises and transfers the caller to the
    # owner, inside the transfer hours. False = it asks the caller to call again, and hangs up.
    transfer_on_failure: bool = True

    # Text messages (off until the number has a Telnyx messaging profile and a 10DLC campaign).
    sms_enabled: bool = False
    portfolio_url: str = ""  # the link an agent can text to a caller
    website_url: str = ""  # the owner's public website; every agent says it on request, and the goodbye names it

    # The owner's published writing (an RSS feed). The profile agent knows the titles and searches the text.
    writing_feed_url: str = ""
    writing_cache_path: Path = Path("data/writing.json")  # the copy used when the feed cannot be read
    writing_embedding_model: str = "BAAI/bge-small-en-v1.5"  # local, via fastembed; "" = keyword search only
    writing_model_dir: Path = Path("data/models")  # the embedding model is downloaded here once

    # Number lookup on each new caller (Telnyx bills each lookup; results are cached per number).
    caller_lookup: Literal["off", "carrier", "carrier+name"] = "carrier+name"

    # Public HTTPS URL of this server (Tailscale Funnel or ngrok), no trailing slash.
    public_url: str = ""
    port: int = 8000
    # The address the server listens on. On the laptop the tunnel reaches 127.0.0.1. In a container
    # (Fly.io) the host's proxy comes from outside the container, so the image sets 0.0.0.0.
    host: str = "127.0.0.1"

    # Voice legs
    openai_api_key: SecretStr = SecretStr("")
    openai_realtime_model: str = "gpt-realtime-2.1-mini"
    openai_realtime_voice: str = "marin"  # fallback when an agent spec names no voice
    # Turn detection decides how fast the model answers after the caller stops talking.
    # "server": fixed silence timer (fastest). "semantic": the model judges if the caller is done.
    openai_turn_detection: Literal["server", "semantic"] = "server"
    openai_vad_silence_ms: int = 400  # "server" only: silence before the turn ends
    # "server" only: 0.0-1.0; higher ignores more background noise. At 0.6 a 0.2 s noise burst counted as a
    # caller turn, and the agent answered it (the real call of 2026-11-17 00:42). At 0.8 it does not, a
    # caller at -33 dBFS is still heard, and the answer is no slower (simulated calls).
    openai_vad_threshold: float = 0.8
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

    # Amazon Nova Sonic voice leg, on Amazon Bedrock. The names start with NOVA_ on purpose: the
    # standard AWS_ variables in a shell can belong to another AWS account, and must not pay for calls.
    # Without the two keys, agents set to nova_sonic run on another provider.
    nova_aws_access_key_id: str = ""
    nova_aws_secret_access_key: SecretStr = SecretStr("")
    nova_aws_region: str = "us-east-1"  # Nova 2.5 Sonic: us-east-1, us-west-2, eu-north-1, ap-northeast-1
    nova_sonic_model: str = "amazon.nova-2-5-sonic"
    nova_endpointing: Literal["LOW", "MEDIUM", "HIGH"] = "HIGH"  # how fast Nova decides the caller has stopped; HIGH is fastest

    # Move agents to another voice leg, for example "consultant=nova_sonic,tour=gemini_live".
    # Empty = the defaults in agents/specs.py.
    agent_legs: str = ""

    @property
    def webhook_url(self) -> str:
        return f"{self.public_url.rstrip('/')}/webhooks/telnyx"

    @property
    def hold_audio_url(self) -> str:
        """Where Telnyx fetches the waiting sound; "" when this server has no public URL."""
        if self.transfer_hold_audio_url:
            return self.transfer_hold_audio_url
        return f"{self.public_url.rstrip('/')}/audio/ringback.wav" if self.public_url else ""

    @property
    def media_stream_url(self) -> str:
        base = self.public_url.rstrip("/").replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return f"{base}/media"


@lru_cache
def get_settings() -> Settings:
    return Settings()
