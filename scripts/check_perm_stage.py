"""Montage-reordering check for a v5 package (the path the independent review could not cover): the same sessions
streamed once in the training channel order and once with the channels permuted and announced through ch_names must
give identical labels and routed outputs, with the own-space stage active.
Usage: python scripts/check_perm_stage.py PKG [N_SESSIONS=4]"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402


def load(folder, names=None):
    spec = importlib.util.spec_from_file_location("pkg", folder / "submission.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    meta = {"submission_dir": folder};
    if names is not None:
        meta["ch_names"] = names
    return mod.Solver.__new__(mod.Solver).load_model(meta)


def stream(m, X, st, rows):
    rec, staged, step0 = [], [], m.route.step

    def step(*a):
        o = step0(*a); rec.append(np.array(o)); staged.append(m.route.stage is not None); return o
    m.route.step = step; labels = []
    for i, j in enumerate(rows):
        if i == 0 or st[j] != st[rows[i - 1]]:
            m.reset_state()
        labels.append(int(np.asarray(m.predict(X[j:j + 1]))[0]))
    return np.array(labels), np.array(rec), np.array(staged)


def main():
    pkg = Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 4; d = protocol.load(); st = d["stream"]
    sess = list(dict.fromkeys(st)); pick = [sess[i] for i in np.linspace(0, len(sess) - 1, n).astype(int)]; rows = np.flatnonzero(np.isin(st, pick))
    names = [str(c) for c in d["ch_names"]]; perm = np.random.default_rng(0).permutation(len(names))
    la, ra, sa = stream(load(pkg), d["X"], st, rows)
    m = load(pkg, [names[i] for i in perm]); assert m.perm is not None, "the permuted montage was not detected"
    lb, rb, sb = stream(m, d["X"][:, perm], st, rows)
    out = dict(sessions=[s.split("/")[-1] for s in pick], windows=int(len(rows)), staged_windows=int(sa.sum()), labels_equal=bool((la == lb).all()), staged_equal=bool((sa == sb).all()),
               max_output_diff=float(np.abs(ra - rb).max()))
    print("RESULT", json.dumps(out)); assert out["labels_equal"] and out["staged_equal"] and out["max_output_diff"] < 1e-9 and out["staged_windows"] > 0


if __name__ == "__main__":
    main()
