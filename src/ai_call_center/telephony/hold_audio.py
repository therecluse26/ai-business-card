"""The sound a caller hears while the owner's phone rings: a US ring tone, made here.

Telnyx plays it from a URL on this server, in a loop. A tone made in code has no licence to
check. `TRANSFER_HOLD_AUDIO_URL` replaces it with any WAV or MP3 file, for example hold music.
"""

from __future__ import annotations

import io
import wave
from functools import lru_cache

import numpy as np

RATE = 8000  # Hz; the phone network carries no more
TONE_SECS, PAUSE_SECS = 2.0, 4.0  # the US ring tone: 2 s of 440 Hz + 480 Hz, then 4 s of silence
LEVEL = 0.09  # each of the two tones, of full scale: about -21 dBFS together, near the agents' speech level


@lru_cache(maxsize=1)
def ringback_wav() -> bytes:
    """One ring and one pause as a WAV file: 8 kHz, 16-bit, mono, 6 seconds."""
    t = np.arange(int(RATE * TONE_SECS)) / RATE
    tone = LEVEL * (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 480 * t))
    fade = int(RATE * 0.01)  # 10 ms in and out, so the tone does not click
    tone[:fade] *= np.linspace(0.0, 1.0, fade)
    tone[-fade:] *= np.linspace(1.0, 0.0, fade)
    samples = np.concatenate([tone, np.zeros(int(RATE * PAUSE_SECS))])
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes((samples * 32767).astype("<i2").tobytes())
    return out.getvalue()
