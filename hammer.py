# hammer.py — ピアノ合成のためのハンマーと弦の接触モデル
# -----------------------------------------------------------
# このモジュールは MIDI ベロシティを、物理モデルに基づいた
# 弦への励振（初期入力）に変換します。
# RMS値などの後処理に頼らず、
# ハンマー質量と衝突速度だけで振幅を決定します。
# -----------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = ["Hammer","velocity_to_mps",]

# -----------------------------------------------------------------------------
# 定数定義：グランドピアノの典型的なハンマー特性に基づく
# -----------------------------------------------------------------------------
_HAMMER_MASS_KG: Final = 0.003        # ハンマー質量（kg）：3g程度の中音域ハンマー
_MAX_MIDI_VELOCITY: Final = 127       # 最大MIDIベロシティ値
_MAX_HAMMER_SPEED_MPS: Final = 4.0    # 最大衝突速度（m/s）：MIDI127時に約4m/s
_ALPHA: Final = 3.0                   # 非線形ばね指数（α）フェルトの硬さに相当
_STIFFNESS: Final = 1.0e6             # 接触ばね定数（N/m^α）：概算値
_FS: Final = 44_100                   # サンプリング周波数（Hz）

# -----------------------------------------------------------------------------
# 補助関数：MIDIベロシティ（0〜127）→ m/s に変換
# -----------------------------------------------------------------------------

def velocity_to_mps(velocity: int, v_max: float = _MAX_HAMMER_SPEED_MPS) -> float:
    """MIDIベロシティ (0–127) を物理的なハンマー速度 [m/s] に変換する。"""
    velocity = max(0, min(_MAX_MIDI_VELOCITY, velocity))  # 範囲内にクランプ
    return v_max * velocity / _MAX_MIDI_VELOCITY          # 線形マッピング

# -----------------------------------------------------------------------------
# メインクラス：Hammer
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class Hammer:
    """質量-ばね接触に基づいたシンプルなハンマーモデル。

    Parameters
    ----------
    mass : ハンマー質量 [kg]
    stiffness : 接触ばね定数 k [N/m^α]
    alpha 非線形ばねの指数（felt：おおよそ3）
    fs  サンプリング周波数（力プロファイル生成用）
    """

    mass: float = _HAMMER_MASS_KG
    stiffness: float = _STIFFNESS
    alpha: float = _ALPHA
    fs: int = _FS

    # ------------------------------------------------------------------
    # エネルギー → 振幅変換（物理モデルに基づく）
    # ------------------------------------------------------------------
    def energy(self, v_mps: float) -> float:
        """衝突速度 v に対する運動エネルギー [J] を返す。"""
        return 0.5 * self.mass * v_mps**2

    def velocity_to_amplitude(self, velocity: int) -> float:
        """MIDIベロシティ → 弦の初期振幅（0〜1、線形）

        velocity=127 で 振幅 ≈ 1.0 になるように正規化。
        エネルギーの平方根（√E）を取ることで、音量の知覚が線形に近づくよう調整。
        """
        v = velocity_to_mps(velocity)                    # 速度（m/s）に変換
        e_norm = self.energy(_MAX_HAMMER_SPEED_MPS)      # 最大エネルギー
        amp = math.sqrt(self.energy(v) / e_norm)         # エネルギー比率の平方根
        return min(max(amp, 0.0), 1.0)                    # 範囲 [0,1] に収めて返す

    # ------------------------------------------------------------------
    # 力プロファイル（接触時の外力変化をオプションで出力）
    # ------------------------------------------------------------------
    def force_profile(self, duration: float, velocity: int) -> np.ndarray:
        """接触期間中の力のプロファイル [N] を raised-cosine で生成する。"""
        n = max(1, int(duration * self.fs))  # サンプル数
        t = np.linspace(0.0, duration, n, endpoint=False)  # 時間軸

        # ハンマー質量×速度から最大変位を推定（ばねモデル簡略化）
        v = velocity_to_mps(velocity)
        x_peak = (self.mass * v**2 / self.stiffness) ** (1.0 / (self.alpha + 1))
        f_peak = self.stiffness * x_peak**self.alpha  # 最大力 [N]

        # Raised cosine 曲線で力の時間変化を生成（0→f_peak→0）
        profile = 0.5 * f_peak * (1.0 - np.cos(2 * np.pi * t / duration))
        return profile.astype(np.float32)
