#!/usr/bin/env python3
"""
レンダリングスクリプト:
    python render_notes.py 60 108 0.178
    # ないし
    python render_notes.py        # サンプルセットを一気に出力
"""
import sys, itertools
from pathlib import Path
from modules.synth import synth_note
from modules.io import save_wav


def render(note, vel, gate):
    y = synth_note(note, vel, gate)
    fname = Path(f"orig_wav/n{note}_v{vel}_g{int(gate*1000)}.wav")
    save_wav(y, fname)
    print(f"✔ saved {fname}  ({len(y)/44100:.2f}s)")

if len(sys.argv) == 4:
    n, v, g = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    render(n, v, g)
else:
    # notes = [100, 75, 50, 25]           
    notes = [100,25]                 
    velocities = [120,40]            
    # gate = [0.1, 0.3, 1.0, 3.0]
    gate = [1.0, 3.0]
    for n, v, g in itertools.product(notes, velocities, gate):
        render(n, v, g)
