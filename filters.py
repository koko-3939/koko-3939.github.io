# filters.py — Basic biquad low‑pass / high‑pass utilities
# ---------------------------------------------------------
# Stand‑alone DSP helpers (no external libs). Coefficients are
# calculated at runtime; processing is per‑sample for clarity and easy
# autograd replacement if needed.
# ---------------------------------------------------------
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Final

import numpy as np

_FS: Final = 44_100

__all__ = ["BiquadLP", "BiquadHP"]


def _coeff_lp(fc: float, fs: int) -> tuple[float, float, float]:
    w0 = 2 * math.pi * fc / fs
    alpha = math.sin(w0) / 2 * math.sqrt(2)
    cos_w0 = math.cos(w0)
    b0 = (1 - cos_w0) / 2
    b1 = 1 - cos_w0
    b2 = (1 - cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return (b0 / a0, b1 / a0, b2 / a0), (a1 / a0, a2 / a0)


def _coeff_hp(fc: float, fs: int) -> tuple[float, float, float]:
    w0 = 2 * math.pi * fc / fs
    alpha = math.sin(w0) / 2 * math.sqrt(2)
    cos_w0 = math.cos(w0)
    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = (1 + cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return (b0 / a0, b1 / a0, b2 / a0), (a1 / a0, a2 / a0)


@dataclass(slots=True)
class BiquadLP:
    fc: float
    fs: int = _FS
    z1: float = 0.0
    z2: float = 0.0

    def __post_init__(self):
        self.b, self.a = _coeff_lp(self.fc, self.fs)

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        b0, b1, b2 = self.b
        a1, a2 = self.a
        z1 = self.z1
        z2 = self.z2
        for i, xi in enumerate(x):
            yi = b0 * xi + z1
            z1_new = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            z1 = z1_new
            y[i] = yi
        self.z1, self.z2 = z1, z2
        return y.astype(np.float32)


@dataclass(slots=True)
class BiquadHP:
    fc: float
    fs: int = _FS
    z1: float = 0.0
    z2: float = 0.0

    def __post_init__(self):
        self.b, self.a = _coeff_hp(self.fc, self.fs)

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        b0, b1, b2 = self.b
        a1, a2 = self.a
        z1 = self.z1
        z2 = self.z2
        for i, xi in enumerate(x):
            yi = b0 * xi + z1
            z1_new = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            z1 = z1_new
            y[i] = yi
        self.z1, self.z2 = z1, z2
        return y.astype(np.float32)





class AllpassFDN:
    """4-tap unit-gain feedback delay network (Schroeder style)."""

    def __init__(self, delays: list[int] = [101, 143, 165, 177], g: float = 0.7):
        self.delays = delays
        self.g = g
        self.buffers = [np.zeros(d, dtype=np.float32) for d in delays]
        self.indices = [0]*4

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x, dtype=np.float32)
        for n, xn in enumerate(x):
            acc = xn
            for i in range(4):
                buf, idx = self.buffers[i], self.indices[i]
                out = buf[idx]
                buf[idx] = xn + (-self.g) * out
                self.indices[i] = (idx + 1) % len(buf)
                acc += self.g * out
            y[n] = acc * 0.25  # normalise
        return y
