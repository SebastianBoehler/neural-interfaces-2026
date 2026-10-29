"""Shin 2017 A + B as one external corpus with PROTEUS labels (R13).

The same 29 participants did motor imagery (A: left/right hand) and mental
arithmetic (B) in alternating sessions of one recording day. Merging each
participant's A and B windows into one session keeps the MI-versus-
arithmetic contrast through per-session alignment, unlike single-task
corpora (Dreyer 2023), where alignment erases the task mean.

Labels: hand imagery -> 0 (PROTEUS motor imagery), subtraction -> 1
(PROTEUS mental subtraction); rest windows are dropped.
Input: data/shin2017a/windows_*.npz, data/shin2017b/windows_*.npz
(scripts/export_moabb.py). Output: data/shin2017ab/windows.npz, read by
protocol.load(external=["shin2017ab"]) (run_cv NAME@shin2017ab).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("X", "y", "subject", "session", "run", "onset")


def load(name):
    parts = []
    for f in sorted((ROOT / "data" / name).glob("windows_*.npz")):
        with np.load(f) as z:
            classes = [str(c) for c in z["classes"]]
            lab = np.array([0 if "hand" in c else 1 if "subtraction" in c or "substraction" in c
                            else -1 for c in classes])
            y = lab[z["y"]]
            keep = y >= 0
            part = {k: z[k][keep] for k in KEYS}
            part["y"] = y[keep]
            part["session"] = np.char.add(f"{name[-1]}", part["session"].astype(str))
            part["subject"] = np.char.replace(part["subject"].astype(str), "Shin2017A", "Shin2017AB")
            part["subject"] = np.char.replace(part["subject"], "Shin2017B", "Shin2017AB")
            parts.append(part)
            ch = z["ch_names"]
    return {k: np.concatenate([p[k] for p in parts]) for k in KEYS}, ch


def main():
    a, ch = load("shin2017a")
    b, _ = load("shin2017b")
    d = {k: np.concatenate([a[k], b[k]]) for k in KEYS}
    run = np.char.add(np.char.add(d["session"], "_"), d["run"].astype(str))
    d["run"] = np.unique(run, return_inverse=True)[1].astype(np.int64)  # int like PROTEUS
    order = np.argsort(d["subject"], kind="stable")  # A then B within a participant
    d = {k: v[order] for k, v in d.items()}
    d["session"] = np.full(len(d["y"]), "ab")  # one alignment session per participant
    n = len(d["y"])
    d.update(split=np.full(n, "external"), interface=np.full(n, "Shin2017AB"),
             task=np.full(n, "Shin2017AB"), ch_names=ch)
    out = ROOT / "data/shin2017ab/windows.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **d)
    print(json.dumps({"out": str(out), "windows": n, "subjects": len(set(d["subject"])),
                      "counts": np.bincount(d["y"], minlength=3).tolist()}))


if __name__ == "__main__":
    main()
