# soundboard.py — Piano soundboard & room body resonance model
# -------------------------------------------------------------
# Purpose
#   • Adds realistic broadband resonance and late decay tails that the
#     plain string model lacks.
#   • Implemented as an FIR impulse response built from a small set of
#     damped modes; convolution done with FFT (overlap‑add) for speed.
#   • Mode frequencies are weakly note‑dependent: bass notes excite
#     lower modes relatively more.
# -------------------------------------------------------------
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, List

import numpy as np
from scipy.signal import fftconvolve

__all__ = ["SoundBoard"]

_FS: Final = 44_100


@dataclass(slots=True)
class Mode:
    freq: float  # Hz
    decay: float  # seconds (T60)
    gain: float  # linear gain (per‑mode weight)

    def impulse(self, length: int, fs: int) -> np.ndarray:
        t = np.arange(length) / fs
        env = np.power(10.0, -3 * t / self.decay)  # -60 dB per decay seconds
        return env * np.sin(2 * math.pi * self.freq * t) * self.gain


# ------------------------------------------------------------------
# Pre‑defined modal template (mid‑register)
# ------------------------------------------------------------------
BASE_MODES: List[Mode] = [
    Mode(120.0, 3.0, 0.4),   # soundboard fundamental
    Mode(250.0, 2.8, 0.6),
    Mode(500.0, 2.5, 0.6),
    Mode(1000.0, 2.2, 0.5),
    Mode(2000.0, 1.8, 0.4),
    Mode(3500.0, 1.5, 0.3),
]


# ------------------------------------------------------------------
@dataclass(slots=True)
class SoundBoard:
    fs: int = _FS
    ir_seconds: float = 2.5
    use_fftconv: bool = True

    def _make_impulse(self, note: int) -> np.ndarray:
        """Generate note‑dependent impulse response."""
        length = int(self.ir_seconds * self.fs)
        ir = np.zeros(length, dtype=np.float32)

        # Weight modes depending on note: bass notes excite lower modes more
        note_offset = (note - 60) / 24  # -1 .. +1 approx across range
        for m in BASE_MODES:
            # Gain scaling: ±3 dB across range
            gain = m.gain * (1 + 0.3 * (-note_offset if m.freq < 500 else note_offset))
            ir += Mode(m.freq, m.decay, gain).impulse(length, self.fs)

        # Normalise
        ir /= max(1e-9, np.abs(ir).max())
        return ir

    # --------------------------------------------------------------
    def apply(self, y: np.ndarray, note: int) -> np.ndarray:
        """Convolve *y* with note‑dependent soundboard IR."""
        ir = self._make_impulse(note)
        if self.use_fftconv:
            return fftconvolve(y, ir, mode="full").astype(np.float32)
        else:
            return np.convolve(y, ir, mode="full").astype(np.float32)
        

# Alias for spec sheet naming consistency
SoundBoardFIR = SoundBoard
