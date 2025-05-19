# damper.py — Piano damper / key-off attenuation model
# ------------------------------------------------------
# Purpose
#   • Simulate the felt damper that presses on strings when the key is
#     released (sustain pedal up).
#   • Provides rapid high-frequency attenuation plus overall amplitude
#     decay that depends on note and velocity.
# Usage
#   damper = DamperFilter(fs)
#   y_damped = damper.apply(y, note_number, velocity, pedal=False)
# ------------------------------------------------------
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = ["DamperFilter"]

_FS: Final = 44_100


@dataclass(slots=True)
class DamperFilter:
    """Key-off damper model (first-order low-pass & exponential decay).

    Parameters
    ----------
    fs : int
        Sample rate.
    hf_cut_base : float
        Base high-frequency cut (Hz) for middle C (note 60) at velocity 100.
    time_const_base : float
        Amplitude time constant (seconds) for middle C at velocity 100.
    """

    fs: int = _FS
    hf_cut_base: float = 2_000.0
    time_const_base: float = 0.06  # 60 ms

    # ------------------------------------------------------------------
    # Helper: note-dependent scaling
    # ------------------------------------------------------------------
    def _scale_by_note(self, note: int, base: float, exp: float) -> float:
        """Scale *base* by (f_ratio)**exp where f_ratio = 2**((note-60)/12)."""
        ratio = 2.0 ** ((note - 60) / 12.0)
        return base * ratio ** exp

    # ------------------------------------------------------------------
    def apply(
        self,
        y: np.ndarray,
        note: int,
        velocity: int,
        pedal: bool = False,
    ) -> np.ndarray:
        """Apply damper after key-off.

        Parameters
        ----------
        y : np.ndarray
            Input waveform (float32)
        note : int
            MIDI note (55–94)
        velocity : int
            MIDI velocity (103–112). Faster velocity → stronger initial energy,
            but damper felt enters with similar force; perceived decay slightly
            slower, so we lengthen time constant a bit.
        pedal : bool
            If True (sustain pedal down), **no damping** is applied.
        """
        if pedal:
            return y  # sustain pedal keeps strings free

        # --- compute parameters ---
        hf_cut = self._scale_by_note(note, self.hf_cut_base, 0.5)  # higher notes, higher cut
        tau = self._scale_by_note(note, self.time_const_base, -0.3)  # bass damps slower
        tau *= 1.0 + 0.003 * (velocity - 103)  # velocity 112 ≈ +27% decay time

        # --- low-pass coefficient (one-pole) ---
        rc = 1.0 / (2 * math.pi * hf_cut)
        alpha = 1.0 / (1.0 + rc * self.fs)  # 0<alpha<1

        # --- amplitude envelope ---
        env = np.exp(-np.arange(len(y)) / (tau * self.fs))
        y_out = np.empty_like(y)

        lp_state = 0.0
        for i, s in enumerate(y * env):
            lp_state += alpha * (s - lp_state)
            y_out[i] = lp_state

        return y_out.astype(np.float32)
