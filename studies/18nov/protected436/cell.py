"""Run the private solver on raw windows and check fixed same-window endpoint controls."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from returning_pair_cell import inputs
from returning_pair_guard import BUNDLE, ROOT, fresh_model
from returning_assets import sha
from package_guard import instrument
from record import ft_clock_restart, load_model

OLD = ROOT / "ni26_js321_20261117_returning_pair_cpu1"
OUT = Path("/out")


def reset(model):
    model = fresh_model(model)
    torch.manual_seed(3211701)
    ft_clock_restart(model)
    if model.recB or model.cp.t or model.cf.dead:
        raise ValueError("session reset failed")
    return model


def predict(model, raw):
    return int(model.predict(torch.from_numpy(raw[None]))[0])


def unknown(X, channels, fold, rows):
    cases = []
    for path, name in [(BUNDLE / fold / "js", "js"), (Path("packages") / fold, "protected")]:
        model = instrument(load_model(path, channels, name, "cpu"), True)
        model = reset(model)
        scores, labels = [], []
        for row in rows:
            labels.append(predict(model, X[row]))
            scores.append((model.cp._observed if name == "js" else model.output436).copy())
        if model.recB or model.cp.fused < 1 or model.ft.refits < 1:
            raise ValueError("forced unknown control did not exercise CP and FT")
        cases.append((scores, labels))
    if not all(np.array_equal(a, b) for a, b in zip(*cases)):
        raise ValueError("forced unknown runtime differs from JS")
    return dict(windows=len(rows), refits=model.ft.refits, fused=model.cp.fused)


def main(index):
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized():
        raise RuntimeError("expected pinned CPU worker")
    fold, order, X, channels, canonical, source, delivery, sessions, views, _, keys = inputs(max(0, index))
    if index == -1:
        result = unknown(X, channels, fold, np.resize(source, 100))
        (OUT / "proof.json").write_text(json.dumps(result))
        return
    old = OLD / f"cell_{index}"
    proof = json.loads((old / "proof.json").read_text())
    if proof["predictions_sha256"] != sha(old / "predictions.npz") or proof["source_sha256"] != sha(OLD / "source.sha256"):
        raise ValueError("control source or bank hash differs")
    with np.load(old / "predictions.npz") as z:
        bank = dict(z)
    if not np.array_equal(bank["index"], canonical) or not np.array_equal(bank["base_scores"], bank["off_scores"]):
        raise ValueError("control row identity or off endpoint differs")
    target = Path("packages") / fold
    template = load_model(target, channels, "protected", "cpu")
    for row in source:
        predict(template, X[row])
    template.reset_state()
    labels = np.full(len(canonical), -1)
    records = []
    for session in dict.fromkeys(sessions[delivery]):
        model = reset(template)
        rows = delivery[sessions[delivery] == session]
        counts = [0, 0]
        for row in rows:
            pos = int(np.searchsorted(canonical, row))
            label = predict(model, X[row])
            arm = "base" if model.recB else "js"
            counts[int(model.recB)] += 1
            if label != bank[arm + "_labels"][pos] or not np.array_equal(model.output436, bank[arm + "_scores"][pos]):
                raise ValueError(f"runtime/control difference at {fold} {order} {row} {arm}")
            if not np.array_equal(model.trace["u"], views[pos]):
                raise ValueError("pinned order/view differs")
            labels[pos] = label
        historical = next(r["activation"]["base"] for r in proof["records"] if r["session"] == session)
        if not all(counts) or model.log6["rec"] != historical["rec"] or model.base_personal.stopped != historical["cp_stopped"]:
            raise ValueError("actual recognition or protected CP differs")
        records.append(dict(session=str(session), js=counts[0], baseline=counts[1]))
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("incomplete runtime output")
    np.savez(OUT / "runtime.npz", index=canonical, labels=labels)
    result = dict(index=index, fold=fold, order=order, keys=keys, records=records, bitwise=True,
                  source=sha("source.sha256"), package=sha(target / "submission.py"),
                  runtime=sha(OUT / "runtime.npz"), seconds=time.monotonic()-start)
    (OUT / "proof.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main(int(sys.argv[1]))
