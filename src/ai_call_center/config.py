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
from pydantic import SecretStr, field_validator
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
    # The grader: after hangup, this text model scores the call against the model rules in grading.py.
    # Empty = the code rules only. The scores go to the analytics store (analytics.py).
    grader_model: str = "anthropic:claude-sonnet-5-5"
    # The decision grader (decision_grader.py): a decision model that hears the audio of a voice call and answers
    # typed questions. It needs GOOGLE_API_KEY. For a voice call it is the first choice, and GRADER_MODEL reads
    # the call only when the decision model gives no answer. Empty = off.
    grader_decision_model: str = "gemini-decision-1"
    grader_hears_call: bool = True  # false = the call audio stays on the server, and the decision model reads the transcript
    grader_fail_at: float = 0.8  # a rule fails only at this probability of a failure, or more
    # true = the grader model checks each voice call too, and the log shows where the two differ. It is one more
    # request for each call. Set it to false when the two agree on real calls.
    grader_compare: bool = True
    # The git commit of this build, for the analytics. The container image sets it; empty = ask git.
    harness_version: str = ""
    # After each call, the public report (analytics.public_report) goes to this Cloudflare R2 bucket as one
    # JSON file (publish.py). With one of the first four empty, there is no upload.
    report_r2_account_id: str = ""
    report_r2_bucket: str = ""
    report_r2_access_key_id: str = ""
    report_r2_secret_access_key: SecretStr = SecretStr("")
    report_r2_key: str = "calls.json"  # the name of the file in the bucket
    # A fix request starts the job on GitHub that proposes a fix for a failed rule (`improve.py`).
    # With the token or the repository empty, no fix request is sent.
    improve_github_token: SecretStr = SecretStr("")  # a fine-grained token: this repository only, "Actions" (write) only
    improve_repo: str = ""  # "owner/name"
    improve_workflow: str = "improve.yml"
    improve_ref: str = "main"
    # A caller's feedback becomes a GitHub issue (feedback.py). It uses the token above, which then needs "Issues" (read and write) too.
    feedback_issues: bool = True
    # After a fix is deployed, the server checks that the rule fails less often (fix_check.py). It uses the token above,
    # which then needs "Contents" (read) and "Pull requests" (read) too. A failed check opens an issue: "Issues" (write).
    fix_check: bool = True
    fix_check_calls: int = 5  # graded calls after the merge, over all versions that hold the fix, before the check runs
    # At the start of a call and after a saved message, the agent offers a passphrase. With it, the agents may use the notes of that call on the caller's next call (passphrase.py).
    caller_passphrase_offer: bool = True
    owner_name: str = "the owner"
    owner_name_pronunciation: str = ""  # optional; how the agents say the owner's name, in plain words
    # Words written so that a voice model says them right, as "<written>=<respelling>" pairs with commas
    # between: "Magyar=Mag-yar". The voice models read the respelling; people read the written word.
    # See voice/respell.py. This is stronger than the hint above: a voice model says what it reads.
    respellings: str = ""
    # Play the stored recording of the greeting (voice/stored_greeting.py) when one was made for exactly
    # this greeting, leg and voice: the first word comes at once, and the owner's name is always right.
    # Without a matching recording, or when this is off, the greeting agent's model says the greeting.
    stored_greeting: bool = True
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

    # Jev, the decision model of TypeSafe AI (see decisions.py). It reads a caller's request for a live
    # transfer for its meaning, and gives the probability of a yes. Empty key = the word list decides.
    typesafe_api_key: SecretStr = SecretStr("")
    jev_model: str = "jev-latest"
    jev_timeout_ms: int = 500  # after this the word list's answer stands; measure with `uv run check-jev`

    # The owner's CRM (zepto-crm), reached over MCP (see crm.py). After a call that left a message, code writes the
    # caller, the call, a deal and a task. With the address or the token empty, nothing is written or read.
    crm_mcp_url: str = ""  # https://<crm-host>/ext/mcp
    crm_api_token: SecretStr = SecretStr("")  # an API token of the CRM, scope "full"; make it from a user with the role member
    crm_timeout_secs: float = 10.0  # for one request; check the real times with `uv run check-crm`

    # A short meeting with the owner, booked during the call on Cal.com (see meetings.py). Empty key = off, and
    # no agent says anything about a meeting. The event type is found by its slug, so no id of an account is here:
    # "15min" is the event type that every new Cal.com account has.
    calcom_api_key: SecretStr = SecretStr("")
    calcom_event_type_slug: str = "15min"
    # All callers together, in 24 hours. A booking makes Cal.com email an invitation to the address a caller
    # gives, so this bounds what a bad actor can send to a stranger. 0 = no booking. One call books one at most.
    calcom_max_bookings_per_day: int = 3
    calcom_timeout_secs: float = 5.0  # for one request; the caller waits for it

    # Pipecat Voice Reader (see voice/voice_reader.py): a small local model that hears the sound of each
    # caller turn: tone of voice, laughter, speech rate. It reads the sound only, not the words.
    voice_reader: bool = True
    voice_reader_variant: Literal["base", "tiny"] = "base"  # "tiny" costs about half the CPU and reads tone less well
    voice_model_dir: Path = Path("data/models")  # the model is downloaded here once, about 28 MB
    voice_steer: bool = True  # when the caller sounds frustrated, the agent gets one note (OpenAI agents only)

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

    # Phonic voice leg. No agent runs on it by default: it bills each open minute. Without PHONIC_API_KEY,
    # agents set to phonic run on another provider.
    phonic_api_key: SecretStr = SecretStr("")
    phonic_model: str = ""  # empty = Phonic's default, which reported itself as phonic-1 on 2026-11-24
    # How long Phonic waits before it decides the caller has stopped talking. "standard" is its default.
    phonic_turn_taking: Literal["eager", "standard", "relaxed", "measured", "patient", "very_patient"] = "standard"

    # Move agents to another voice leg, for example "consultant=nova_sonic,tour=gemini_live".
    # The greeting agent on another leg: "classifier=openai_realtime" or "classifier=phonic".
    # Empty = the defaults in agents/specs.py.
    agent_legs: str = ""

    @field_validator("respellings")
    @classmethod
    def _respellings_are_pairs(cls, value: str) -> str:
        from ai_call_center.voice.respell import Respelling

        Respelling.parse(value)  # a typing mistake stops the server at start, not in the middle of a call
        return value

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
