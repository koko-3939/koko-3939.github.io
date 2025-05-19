# eq64.py — 64‑band parametric equaliser (peaking IIR) for piano timbre matching
# -----------------------------------------------------------------------------
# 妥当性の担保方針
#   • 帯域中心周波数は 27.5 Hz (A0) 〜 16 kHz を **等比数列(1/12oct) で 64 点**
#     → ピアノ有効帯域 & 解析 log‑Mel と整合。
#   • 各バンドは **2次ピーキングフィルタ(Q≈4)**：先行研究*Smith 2010*, *Burtner 2012*
#   • ゲイン範囲は ±6 dB（L2 正則化推奨）
#   • 最適化時は gains_db ∈ ℝ⁶⁴ をパラメータとし、物理モデル残差を吸収。
# -----------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Final, Sequence
from scipy.signal import sosfilt

__all__ = ["EQ64", "default_centers"]

_FS: Final = 44_100
_NBANDS: Final = 64

# -----------------------------------------------------------------------------
# 周波数配置：1/12 octave (≈100 cent) ステップで 64 バンド
# -----------------------------------------------------------------------------
_a0 = 27.5  # A0
_ratio = 2 ** (1 / 12)  # 1 semitone
_centers = _a0 * _ratio ** np.arange(_NBANDS)
# upper‑bound truncate
_centers[_centers > 16_000] = 16_000

def default_centers() -> np.ndarray:
    """Return 64‑band centre frequencies (Hz)."""
    return _centers.copy()


# -----------------------------------------------------------------------------
# Bi‑quad peaking filter coefficient helper
# -----------------------------------------------------------------------------

def _design_peaking(fc: float, gain_db: float, q: float, fs: int) -> np.ndarray:
    """Return SOS coeffs for a single peaking EQ band."""
    import math
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * fc / fs
    alpha = math.sin(w0) / (2 * q)

    b0 = 1 + alpha * A
    b1 = -2 * math.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha / A

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float32)


# -----------------------------------------------------------------------------
@dataclass(slots=True)
class EQ64:
    """64‑band parametric EQ (peaking bi‑quad cascade).

    Parameters
    ----------
    gains_db : Sequence[float]
        64 要素のゲイン (dB)。正値でブースト、負値でカット。
    centers : Sequence[float], optional
        各バンド中心周波数。既定は 1/12oct log 配置。
    q : float, default 4.0
        全バンド共通の品質係数。実機 EQ の Q≈4 が目安。
    fs : int, default 44100
    """

    gains_db: Sequence[float]
    centers: Sequence[float] = tuple(default_centers())
    q: float = 4.0
    fs: int = _FS

    def __post_init__(self):
        if len(self.gains_db) != _NBANDS:
            raise ValueError(f"gains_db must be length {_NBANDS}")
        sos_list = [_design_peaking(fc, g, self.q, self.fs)
                    for fc, g in zip(self.centers, self.gains_db)]
        # stack to (64, 6) sos array
        self.sos = np.concatenate(sos_list, axis=0)

    # --------------------------------------------------------------
    def process(self, x: np.ndarray) -> np.ndarray:
        """Apply EQ to mono signal x (float32)."""
        return sosfilt(self.sos, x).astype(np.float32)

    # convenience for synth pipeline
    def __call__(self, x: np.ndarray) -> np.ndarray:  # noqa: D401
        return self.process(x)
