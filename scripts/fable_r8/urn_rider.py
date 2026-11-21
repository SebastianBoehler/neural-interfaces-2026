"""Replay the existing §228c CountFilter at the recorded quarter terminal posterior."""

import argparse
import json
from pathlib import Path

import numpy as np

from detector import digest, fold_detector, signed_hss
from protocol import DATA_SHA
from score_audit import collect
from tapes import protocol


class TerminalUrn:
    def __init__(self, data_path, fixtures, hss):
        if digest(data_path) != DATA_SHA:
            raise ValueError("expected signed EEG")
        self.module = signed_hss(hss)
        self.data = protocol.load(data_path)
        self.windows = self.module.to_windows(self.data.pop("X"), self.module.channel_map(self.data["ch_names"]))
        self.fixtures = fixtures
        self.fold = self.model = None

    def __call__(self, fold, rows, scores, expected_latch):
        if self.fold != fold:
            self.fold = fold
            self.model = fold_detector(self.module, self.fixtures, fold)
        cf = self.model
        cf.reset_state()
        posterior = np.stack([cf.step(self.windows[row], p) for row, p in zip(rows, scores, strict=True)])
        age = -1 if cf.latch is None else cf.latch + 1
        if (age, cf.t, cf.dead, cf.log["err"]) != (expected_latch, len(rows), False, 0):
            raise ValueError(f"terminal CountFilter route differs: {fold}, {age}, {cf.log}")
        before = len(rows) if age < 0 else age - 1
        np.testing.assert_array_equal(posterior[:before], scores[:before])
        return posterior.argmax(1).astype(np.int8)


def smoke(urn, captures, detector):
    fold, order, key = "new_P01", 72000, "72000_1"
    path = captures / f"{fold}-{order}.npz"
    with np.load(path, allow_pickle=False) as tape:
        rows, scores = tape[f"rows_{key}"], tape[f"terminal_{key}"]
        age = int(tape[f"latch_{key}"])
        if age != detector[fold][key] or not 0 < age <= 60:
            raise ValueError("urn smoke did not exercise the registered latch")
        labels = urn(fold, rows, scores, age)
        np.testing.assert_array_equal(labels[:age-1], tape[f"quarter_{key}"][:age-1])
        prefix = urn(fold, rows[:age+5], scores[:age+5], age)
        np.testing.assert_array_equal(labels[:len(prefix)], prefix)
        if np.array_equal(labels, tape[f"quarter_{key}"]):
            raise ValueError("urn smoke did not change any terminal decision")
    print(f"PASS terminal urn: {len(rows)} real predictions, active latch {age}, identical prefix and reset")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "fixtures", "hss", "captures", "detector", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    urn = TerminalUrn(a.data, a.fixtures, a.hss)
    detector = json.loads(a.detector.read_text())
    if a.smoke:
        smoke(urn, a.captures, detector)
    else:
        stats = collect(a.captures, a.data, ("within-fit", "within-held"), detector=detector, urn=urn)
        with a.output.open("x") as f:
            json.dump(stats, f, indent=2)
            f.write("\n")
        print(f"Collected {len(stats['shards'])} terminal-urn tapes")
