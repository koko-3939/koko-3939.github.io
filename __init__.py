# modules/__init__.py
from .hammer import Hammer
from .stringsynth import KSPhysical
from .damper import DamperFilter
from .soundboard import SoundBoard, SoundBoardFIR  # alias
from .filters import BiquadLP, BiquadHP, AllpassFDN
from .utils import midi2freq, db_to_lin, lin_to_db, oversample, round_gate, GATE_BINS

__all__ = [
    "Hammer", "KSPhysical", "StretchedString",
    "DamperFilter", "SoundBoard", "SoundBoardFIR",
    "AmplitudeEnvelope", "BiquadLP", "BiquadHP", "AllpassFDN",
    "midi2freq", "db_to_lin", "lin_to_db", "oversample",
    "round_gate", "GATE_BINS"
]
