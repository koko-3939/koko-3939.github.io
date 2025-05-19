# io.py — lightweight audio I/O helpers
# --------------------------------------
# Focused on quick prototyping and unit–test friendly functions. No heavy
# GUI; playback is optional (requires sounddevice).  All paths and
# filenames are left to caller.
# --------------------------------------
from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import soundfile as sf

try:
    import sounddevice as sd  # optional runtime dependency
except ImportError:  # pragma: no cover
    sd = None  # type: ignore

__all__ = ["save_wav", "play_audio"]


# ------------------------------------------------------------------
# WAV save helper
# ------------------------------------------------------------------

def save_wav(data: np.ndarray, path: str | pathlib.Path, fs: int = 44_100) -> None:
    """Save *data* to 16‑bit PCM WAV.

    Parameters
    ----------
    data : np.ndarray
        Audio signal in range [-1, 1]. Clips are hard‑clamped.
    path : str | Path
        Output file path.
    fs : int, default 44100
        Sample rate.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # 自動でフォルダ作成

    # -1.0…1.0 → int16 へ量子化（クリッピングも兼ねる）
    data16 = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    sf.write(path, data16, fs, subtype="PCM_16")


# ------------------------------------------------------------------
# Quick playback (optional)
# ------------------------------------------------------------------

def play_audio(data: np.ndarray, fs: int = 44_100, block: bool = True) -> Optional[int]:
    """Play *data* via sounddevice if available. Returns stream id or None."""
    if sd is None:
        print("[io] sounddevice not installed → playback skipped")
        return None
    stream = sd.play(data.astype(np.float32), samplerate=fs, blocking=block)
    return stream
