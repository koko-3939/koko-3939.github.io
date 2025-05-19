# utils.py — common helper functions
# -----------------------------------
# • MIDI ↔ frequency conversion
# • dB / linear helpers
# • gate_time quantisation bins for analysis stage
# • value clamping utilities
# -----------------------------------
from __future__ import annotations

import math
import numpy as np
from typing import Final, List
import numpy as np



__all__ = [
    "midi2freq",
    "freq2midi",
    "db_to_lin",
    "lin_to_db",
    "round_gate",
    "clamp",
    "GATE_BINS",
]

_A4_MIDI: Final = 69
_A4_FREQ: Final = 440.0
_LN10_20: Final = 20.0 / math.log(10.0)

# ------------------------------------------------------------------
# MIDI ↔ Hz
# ------------------------------------------------------------------

def midi2freq(note: int) -> float:
    """Convert MIDI note (int) to frequency in Hz."""
    return _A4_FREQ * 2 ** ((note - _A4_MIDI) / 12)


def freq2midi(freq: float) -> float:
    """Convert frequency in Hz to fractional MIDI note."""
    return 12 * math.log2(freq / _A4_FREQ) + _A4_MIDI

# ------------------------------------------------------------------
# dB helpers
# ------------------------------------------------------------------

def db_to_lin(db: float) -> float:
    return 10 ** (db / 20.0)


def lin_to_db(lin: float, eps: float = 1e-12) -> float:
    return 20.0 * math.log10(max(eps, lin))

# ------------------------------------------------------------------
# Gate‑time rounding (analysis only)
# ------------------------------------------------------------------

GATE_BASE: Final = 0.178
GATE_BINS: List[float] = [GATE_BASE * r for r in (0.5, 0.75, 1.0, 1.25, 1.5)]


def round_gate(gate: float) -> float:
    """Round arbitrary gate_time to nearest analysis bin."""
    return min(GATE_BINS, key=lambda g: abs(gate - g))

# ------------------------------------------------------------------
# Misc util
# ------------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)





def oversample(x: np.ndarray, factor: int = 2) -> np.ndarray:
    """Very-lightweight oversampling via linear interpolation."""
    if factor <= 1:
        return x
    n = len(x)
    idx = np.arange(n)
    idx_hi = np.linspace(0, n - 1, n * factor)
    return np.interp(idx_hi, idx, x).astype(np.float32)


def trim_db(y: np.ndarray, db_thresh: float = -80.0) -> np.ndarray:
    """切り落とし：振幅が阈値 (dBFS) 未満になった以降を削る"""
    thr = 10 ** (db_thresh / 20)
    idx = np.where(np.abs(y) > thr)[0]
    return y[: idx[-1] + 1] if idx.size else y


