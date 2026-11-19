"""Example team train.py: python train.py --people P01,... [--first-session-only P04] --out DIR (harness contract)."""
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import data as D  # harness/data.py (PYTHONPATH=/cs/harness)

HERE = Path(__file__).resolve().parent


def main():
    a = D.train_parser()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("sub", HERE / "submission.py")
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    X, y, info = D.load_train(a.people, a.first_session_only, a.data)
    rng = np.random.default_rng(a.seed)
    F = np.zeros((len(y), 41 * 42 // 2))
    stream = np.char.add(info["person"], info["session"])
    for s in dict.fromkeys(stream):  # causal EA over a random delivery order, like the full-session shuffle
        rows = rng.permutation(np.flatnonzero(stream == s))
        al = sub.Aligner()
        for r in rows:
            F[r] = al.step(X[r])
    mu, sd = F.mean(0), F.std(0) + 1e-6
    clf = LogisticRegression(C=0.01, max_iter=500, class_weight="balanced").fit((F - mu) / sd, y)
    np.savez(out / "weights.npz", mu=mu, sd=sd, coef=clf.coef_, b=clf.intercept_)
    shutil.copy(HERE / "submission.py", out / "submission.py")
    print("trained on", len(y), "windows; train acc", clf.score((F - mu) / sd, y))


if __name__ == "__main__":
    main()
