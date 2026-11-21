"""Load each fit-excluded source for the Lane TB-grow live solver."""
import importlib.util
import json
from pathlib import Path

import torch

from verify import B, D, DEVICE, FOLDS, W, WEIGHTS, audit, digest

SELECT_ORDERS = range(71140, 71148)
CONFIRM_ORDERS = range(71148, 71156)
EXTRA_ORDERS = range(71172, 71180)
FINAL_ORDERS = range(71200, 71216)  # finalists head-to-head (21 Nov)
REFERENCE = "nine25"
ARMS = {
    "base": (0.0, 3, "mixer", None),
    "nine25": (0.25, 3, "mixer", None),
    "nine35": (0.35, 3, "mixer", None),
    "nine50": (0.50, 3, "mixer", None),
    "native25": (0.25, 3, "native", None),
    "terminal25": (0.25, 3, "terminal", None),
    "twelve25": (0.25, 4, "mixer", None),
    "eighteen25": (0.25, 6, "mixer", None),
    "bc_a1": (0.25, 3, "mixer", "c_a1"),
    "bc_light": (0.25, 3, "mixer", "c_light"),
    "bc_a2": (0.25, 3, "mixer", "c_a2"),
    "ba_terminal": (0.25, 3, "mixer", "a"),
}
TEAM_A = Path("/leonardo_work/AIFAC_P02_1675/ni26_cleanslate_A2/b6-a1-manas")
TEAM_C = W.parents[2] / "ni26_teamC_20261120/work/C5"
C_MEMBERS = "shallow:0,shallow:1,eegnet:0,eegnet:1,fbnet:0,fbnet:1"
C_STEPS = {"c_a1": 0, "c_light": 1, "c_a2": 4}


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"source anchor count for {old!r}: {text.count(old)}")
    return text.replace(old, new)


def roster(fold, data):
    mode, held = fold.split("_")
    people = D.PEOPLE if mode == "ret" else [p for p in D.PEOPLE if p != held]
    first = [held] if mode == "ret" else []
    rows = D.train_rows(data, people, first)
    return people, first, sorted(set(data["stream"][rows])), len(rows)


def banks(run, fold, data, members):
    original = WEIGHTS / fold / "weights.pt"
    receipt = {"b0": audit(data, fold, original)}
    if members == 3:
        return B.bank(original, 3), receipt
    extra = run / "b" / fold / "weights.pt"
    receipt["b1"] = audit(data, fold, extra)
    old = torch.load(original, map_location="cpu", weights_only=True)
    new = torch.load(extra, map_location="cpu", weights_only=True)
    order = ("shallow",) * 3 + ("slow",) * 3 + ("fbc",) * 3
    for pack in (old, new):
        assert tuple(kind for kind, _ in pack["nets"]) == order
        assert pack["steps"] == 1200 and pack["lam"] == 0.5 and not pack["experts"] and not pack["cfg"]
    k = members - 3
    old["nets"] = [entry for i in (0, 3, 6) for entry in old["nets"][i:i + 3] + new["nets"][i:i + k]]
    for i in (0, 3, 6):
        for (kind, state), (_, extra_state) in zip(
                torch.load(original, map_location="cpu", weights_only=True)["nets"][i:i + k],
                new["nets"][i:i + k], strict=True):
            if all(torch.equal(v, extra_state[name]) for name, v in state.items()):
                raise ValueError(f"duplicate Team B seed for {kind}")
    out = B.io.BytesIO()
    torch.save(old, out)
    return out.getvalue(), receipt


def module(source, variant, text):
    spec = importlib.util.spec_from_file_location(f"lane_tb_grow_{variant}", source)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(text, str(source), "exec"), mod.__dict__)
    return mod


def team_c(run, fold, data, variant):
    pkg = run / "c_e6" / fold if fold.startswith("ret_") else TEAM_C / f"pkg_{fold.split('_')[1]}"
    people, _, streams, n = roster(fold, data)
    training = pkg / "train.json"
    meta = json.loads(training.read_text())
    assert meta["people"] == people and meta["streams"] == streams and meta["rows"] == n
    args = meta["args"]
    assert args["members"] == C_MEMBERS and args["epochs"] == 60 and not args["ext"] and not args["extra"]
    src = pkg / "submission.py"
    mod = module(src, variant, src.read_text())
    model = mod.load_package(pkg, DEVICE, dict(bn="prefix", steps=C_STEPS[variant], balance=0))
    assert len(model.base) == 6 and model.cfg["steps"] == C_STEPS[variant] and model.cfg["ts_beta"] == 0
    return model, dict(weights_sha=digest(pkg / "weights.pt"), source_sha=digest(src), train_sha=digest(training))


def team_a_donor(fold, data, data_sha):
    pkg = TEAM_A / fold
    people, first, _, _ = roster(fold, data)
    training = pkg / "training.json"
    meta = json.loads(training.read_text())
    digest_weights = digest(pkg / "weights.pt")
    assert meta["people"] == people and meta["first_session_only"] == first and meta["public_sha"] == data_sha
    assert meta["weights_sha"] == digest_weights and meta["arch"] == "manas" and meta["recipe"] == "A1"
    assert not meta["smoke"] and len(meta["corpora"]) == 12
    src = pkg / "submission.py"
    return pkg, dict(weights_sha=digest_weights, source_sha=digest(src), train_sha=digest(training))


def team_a(fold, data, data_sha):
    pkg, receipt = team_a_donor(fold, data, data_sha)
    src = pkg / "submission.py"
    original = src.read_text()
    patched = once(original, "        return logits.argmax(-1)",
                   "        self.last = logits.softmax(-1)[0]\n        return logits.argmax(-1)")
    mod = module(src, "manas", patched)
    names = list(data["ch_names"])
    model = mod.Solver().load_model(dict(submission_dir=pkg, device=DEVICE, ch_names=names, n_chans=len(names),
                                        n_times=480, n_classes=3, sfreq=120.0))
    return model, receipt
