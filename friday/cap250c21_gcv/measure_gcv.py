"""Qualify real native covariance reuse and compare warm complete-worker time."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import torch

import gate_rtx
import replay
from live_gate import LiveSelector
from native21 import stream
from scope import external
from settings import sha
from shared_cov import SharedCovSelector


@contextmanager
def with_selector(selector):
    native = gate_rtx.LiveSelector
    gate_rtx.LiveSelector = selector
    try:
        yield
    finally:
        gate_rtx.LiveSelector = native


def paired(template, eeg, rows, receipt):
    source = receipt["unrecognized"]
    warm = stream(template, eeg, rows, "stock", inspect=False)[0]
    sequence = ("stock", "full", "shared", "shared", "full", "stock")
    runs = {name: [] for name in sequence}
    for name in sequence:
        if name == "stock":
            record = stream(template, eeg, rows, "stock", inspect=False)[0]
            assert record["labels"] == warm["labels"]
        else:
            with with_selector(LiveSelector if name == "full" else SharedCovSelector):
                record = gate_rtx.live(template, eeg, rows)[0]
            assert record["labels"] == source["labels"]
            keys = ("rows", "recognized", "eligible", "alternate_refits", "alternate_fused", "alternate_fit_ages")
            assert all(record["alternate"][k] == source["alternate"][k] for k in keys)
            if name == "shared":
                assert record["alternate"]["shared_calls"] == 120 and not record["alternate"]["verified_covariances"]
        assert record["activity"] == warm["activity"] == source["activity"]
        runs[name].append(record)
    stock, full, shared = (np.array([r["seconds"] for r in runs[name]]) for name in ("stock", "full", "shared"))
    return dict(runs=runs, sequence=sequence, share_stock=float(shared.mean()/stock.mean()),
                share_full=float(shared.mean()/full.mean()), worst_stock=float((shared/stock).max()),
                avoided_per_stream=120)


def main(mode):
    assert torch.__version__ == "2.14.1+cu130" and "RTX 2080 Ti" in torch.cuda.get_device_name()
    eeg, who, channels, orders, anchor, archive = external(5)
    template = replay.load(Path("/package"), channels)
    source = sha("source.sha256")
    if mode == "smoke":
        with with_selector(SharedCovSelector):
            proof = gate_rtx.check(template, eeg, who, orders, channels, anchor)
        for name, expected in (("unrecognized", 120), ("known", 10)):
            work = proof[name]["alternate"]
            assert work["shared_calls"] == work["verified_covariances"] == expected
    else:
        receipt = json.loads(Path("smoke.json").read_text())
        assert receipt["source"] == source and receipt["passed"] and not receipt["held_labels_opened"]
        proof = paired(template, eeg, orders[0][who+"|1"], receipt)
    out = Path(f"{mode}.json")
    with out.open("x") as file:
        json.dump(dict(proof, source=source, archive=archive, passed=True, held_labels_opened=False),
                  file, sort_keys=True, allow_nan=False)
    out.chmod(0o444)
    print("PASS F-C21gcv", mode, sha(out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "timing"))
    main(parser.parse_args().mode)
