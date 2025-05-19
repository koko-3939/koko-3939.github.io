# synth.py — high-level API that stitches together the modular piano engine
# -------------------------------------------------------------------------
# Public function
#   synth_note(note_number, velocity, gate_time, **kw) -> np.ndarray
# -------------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

import numpy as np

from modules.hammer      import Hammer
from modules.stringsynth import KSPhysical         
from modules.soundboard  import SoundBoard
from modules.damper      import DamperFilter
from modules.utils       import midi2freq, trim_db, clamp
from modules.eq64        import EQ64   # （使わない場合は import を外しても可）

# ----------------------------------------------------------------------
# デフォルトのシングルトン（差し替え可能）
# ----------------------------------------------------------------------
_hammer  = Hammer()
_string  = KSPhysical()
_board   = SoundBoard()
_damper  = DamperFilter()

# ----------------------------------------------------------------------
# Main API（単音を合成する中核関数）
# ----------------------------------------------------------------------
def synth_note(
    note_number : int,
    velocity    : int,
    gate_time   : float,
    *,
    pedal  : bool = False,
    fs     : int  = 44_100,
    hammer : Hammer        = _hammer,
    string : KSPhysical    = _string,
    board  : SoundBoard    = _board,
    damper : DamperFilter  = _damper
) -> np.ndarray:
    """
    Parameters
    ----------
    note_number : MIDI ノート番号 (25-100 を想定)
    velocity    : 40-120
    gate_time   : ノート ON 持続秒
    pedal       : True=ダンパー解除
    fs          : sample rate
    """

    # ------------------ 入力チェック ------------------
    if not (25 <= note_number <= 100):
        raise ValueError("note_number must be 25-100")
    if not (40 <= velocity <= 120):
        raise ValueError("velocity must be 40-120")
    gate_time = max(0.01, gate_time)

    # ------------------ 1. 基本パラメータ ------------
    f0 = midi2freq(note_number)  

    # ------------------ 2. 弦モデル -------------------
    # 余韻バッファ：RT60 の 2 倍を目安にし、長過ぎる場合は 4 秒でキャップ
    rt60_est  = string._rt60(note_number, velocity)         # 内部メソッドでも利用可
    tail_sec  = min(4.0, 2.0 * rt60_est)

    # 再現性シード：同じ (note, vel, gate) なら常に同じ雑音シード
    seed = ((note_number * 1_000_003) ^ (velocity * 101) ^ int(gate_time * 1e3)) & 0xFFFFFFFF


    raw = string.generate(
        f0,
        gate_time + tail_sec,
        note     = note_number,
        velocity = velocity,
        gate     = gate_time,
        pick_pos = None,                   # デフォルトで OK
        seed     = seed,                    # ← 決定論的シード
        debug    = True
    )

    y = raw                     

    # ------------------ 3. 響板 / ダンパー ------------
    # （必要ならコメントアウトを解除）
    # y = board.apply(y, note_number)
    # y = damper.apply(y, note_number, velocity, pedal=pedal)

    # ------------------ 4. EQ など --------------------
    # eq = EQ64(np.zeros(64))
    # y  = eq.process(y)

    # ------------------ 5. 正規化 ---------------------
    y = trim_db(y, -70.0 if velocity < 60 else -80.0)
    peak = np.max(np.abs(y))
    if peak > 0.99:
        y *= 0.99 / peak

    return y.astype(np.float32)
