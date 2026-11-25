"""A stored recording of the greeting, played the moment the media stream opens.

Every caller hears the same first words (`greeting_words` in agents/specs.py). Asked to say them,
Gemini Live said the owner's surname wrong in about 6% of answers (the first vowel "mah" in place
of "mag"), also with the respelling. And the caller waited for the model's first audio: a median
of about 0.65 s after the request, more on a slow day. A recording of a good take has neither
problem. The greeting agent's session gets the greeting as its own words, without an order to
answer, so it answers the caller's first words as it does today (`seed_without_answer` in legs.py).

A recording is played only when it was made for exactly this greeting: the same spoken words, voice
leg, voice and model. The sidecar file holds a sha256 of each, not the values: the words and the legs
come from `.env`, and no `.env` value goes into the repo. In any other case the model says the
greeting, and the log says why. `uv run record-greeting` makes takes and installs one.
"""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_call_center.agents.specs import AgentSpec, default_voice, greeting_words
from ai_call_center.config import Settings
from ai_call_center.voice.leveler import SPEECH_GATE
from ai_call_center.voice.respell import Respelling

GREETING_WAV = Path(__file__).with_name("greeting.wav")
GREETING_SIDECAR = Path(__file__).with_name("greeting.json")
SAMPLE_RATE = 24_000  # what Gemini Live sends; the output transport brings it to the phone's 8 kHz
TRIM_WINDOW_SECS = 0.01
KEEP_BEFORE_SECS = 0.02  # before the first speech: enough for the start of the first sound, no more
KEEP_AFTER_SECS = 0.15  # after the last speech: the end of the last word fades out


@dataclass(frozen=True)
class StoredGreeting:
    audio: bytes  # PCM 16-bit, mono
    sample_rate: int = SAMPLE_RATE

    @property
    def seconds(self) -> float:
        return len(self.audio) / 2 / max(self.sample_rate, 1)


def spoken_greeting(settings: Settings, respell: Respelling | None = None) -> str:
    """The greeting words as the voice model reads them: with the respellings of RESPELLINGS."""
    return (respell or Respelling.parse(settings.respellings)).spoken(greeting_words(settings.owner_name, settings.brand_name))


def greeting_target(settings: Settings, spec: AgentSpec, respell: Respelling | None = None) -> dict[str, str]:
    """What the greeting agent `spec` (as it runs, after `resolve_spec`) says the greeting with:
    the words, leg, voice and model. The call and `record-greeting` both take them from here."""
    return {"words": spoken_greeting(settings, respell), "leg": spec.leg, "voice": spec.voice or default_voice(spec.name, spec.leg), "model": settings.gemini_live_model}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def greeting_key(words: str, leg: str, voice: str, model: str) -> dict:
    """What a recording was made for: a sha256 of the four values together, and one of each, so the
    log can say which one changed. The values themselves are not kept."""
    parts = {"words": words, "leg": leg, "voice": voice, "model": model}
    return {"sha256": _sha(json.dumps(list(parts.values()))), "parts": {name: _sha(value) for name, value in parts.items()}}


def read_wav(path: Path) -> StoredGreeting:
    """A mono 16-bit PCM WAV file. ValueError for any other kind."""
    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                raise ValueError(f"{path.name} must be mono 16-bit PCM")
            return StoredGreeting(wav.readframes(wav.getnframes()), wav.getframerate())
    except wave.Error as exc:
        raise ValueError(f"{path.name} is not a WAV file: {exc}") from exc


def write_wav(path: Path, greeting: StoredGreeting) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(greeting.sample_rate)
        wav.writeframes(greeting.audio)


def trim_silence(greeting: StoredGreeting) -> StoredGreeting:
    """The take without the silence at its start and end. Each second of silence at the start is a
    second the caller waits for the first word. Quiet means under the voice leveler's speech gate."""
    samples = np.frombuffer(greeting.audio, dtype=np.int16).astype(np.float32) / 32768
    window = max(int(greeting.sample_rate * TRIM_WINDOW_SECS), 1)
    count = len(samples) // window
    if not count:
        raise ValueError("the take has no audio")
    rms = np.sqrt(np.mean(samples[: count * window].reshape(count, window) ** 2, axis=1))
    loud = np.flatnonzero(rms > SPEECH_GATE)
    if not loud.size:
        raise ValueError("the take has no speech")
    start = max(int(loud[0]) * window - int(greeting.sample_rate * KEEP_BEFORE_SECS), 0)
    end = min((int(loud[-1]) + 1) * window + int(greeting.sample_rate * KEEP_AFTER_SECS), len(samples))
    return StoredGreeting(greeting.audio[start * 2 : end * 2], greeting.sample_rate)


def install(take: Path, *, words: str, leg: str, voice: str, model: str, wav: Path = GREETING_WAV, sidecar: Path = GREETING_SIDECAR) -> StoredGreeting:
    """Trim a take and write it, with its sidecar, as the stored greeting."""
    greeting = trim_silence(read_wav(take))
    write_wav(wav, greeting)
    sidecar.write_text(json.dumps({**greeting_key(words, leg, voice, model), "sample_rate": greeting.sample_rate, "seconds": round(greeting.seconds, 2)}, indent=2) + "\n", encoding="utf-8")
    return greeting


def load(words: str, leg: str, voice: str, model: str, *, wav: Path = GREETING_WAV, sidecar: Path = GREETING_SIDECAR) -> tuple[StoredGreeting | None, str]:
    """The stored greeting when it was made for exactly these values. Else (None, why not)."""
    if not wav.exists():
        return None, "no recording is installed (uv run record-greeting)"
    try:
        made_for = json.loads(sidecar.read_text(encoding="utf-8"))
        greeting = read_wav(wav)
    except (OSError, ValueError) as exc:
        return None, f"the recording cannot be read: {exc}"
    want = greeting_key(words, leg, voice, model)
    if not isinstance(made_for, dict) or made_for.get("sha256") != want["sha256"]:
        parts = made_for.get("parts") if isinstance(made_for, dict) and isinstance(made_for.get("parts"), dict) else {}
        changed = [name for name, digest in want["parts"].items() if parts.get(name) != digest]
        return None, f"the recording was made for other {', '.join(changed) or 'settings'}; record it again"
    if greeting.seconds < TRIM_WINDOW_SECS:
        return None, "the recording has no audio"
    return greeting, ""


def choose(*, enabled: bool, fresh: bool, leg_plays: bool, words: str, leg: str, voice: str, model: str, wav: Path = GREETING_WAV, sidecar: Path = GREETING_SIDECAR) -> tuple[StoredGreeting | None, str]:
    """The recording to play as the greeting of this media stream, or (None, why the model says it).

    `fresh`: a new call, whose first words are the standard greeting. A stream that comes back from a
    transfer or a provider change starts with other words, or greets after a failure; the model speaks.
    `leg_plays`: the greeting agent's service can play a recording and take the greeting as its own.
    """
    if not enabled:
        return None, "STORED_GREETING is off"
    if not fresh:
        return None, "this media stream continues a call"
    greeting, why = load(words, leg, voice, model, wav=wav, sidecar=sidecar)
    if greeting is None:
        return None, why
    if not leg_plays:
        return None, f"the greeting agent's leg {leg} cannot play a recording"
    return greeting, ""
