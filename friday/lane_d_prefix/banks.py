"""Pinned corrected Friday inputs and official paired estimates."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from balance import terminal_probability

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data/proteus_windows.npz"
ORDERS = ("full0", "within0")
STOCK = "corrected|cap250"
sys.path.insert(0, str(ROOT / "2026-competition"))
sys.path.insert(0, "/official")
from benchmark_utils.metrics import balanced_accuracy  # noqa: E402


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_manifest():
    for line in Path(__file__).with_name("oracle_inputs.sha256").read_text().splitlines():
        expected, path = line.split()
        if digest(ROOT / path) != expected:
            raise ValueError(f"input differs from preregistration: {path}")


def stock_bank(order, n):
    with np.load(ROOT / f"runs/niv/A_{order}.npz", allow_pickle=False) as bank:
        index, delivery = bank["index"], bank["delivery"]
        raw, terminal = bank[STOCK + "_raw"], bank[STOCK + "_terminal"]
        stock = bank[STOCK]
    with np.load(ROOT / "runs/shuffle_perms.npz", allow_pickle=False) as plan:
        np.testing.assert_array_equal(delivery, plan[f"{order[:-1]}|0|all_loso"])
    np.testing.assert_array_equal(index, np.arange(n))
    np.testing.assert_array_equal(np.sort(delivery), index)
    np.testing.assert_array_equal(terminal.argmax(1), stock)
    probabilities = terminal_probability(terminal)
    np.testing.assert_array_equal(probabilities.argmax(1), stock)
    return delivery, stock, terminal, raw


def metadata(path=DATA):
    with np.load(path, allow_pickle=False) as z:
        subject, session = z["subject"], z["session"]
        visit = np.char.add(np.char.add(subject, "/"), session)
        interface = z["interface"] if "interface" in z.files else np.full(len(subject), "Scherer")
    return subject, visit, np.char.add(np.char.add(visit, "/"), interface)


def score(y, stock, trial, subject, cell):
    """Array shape (order, row). Point estimate and draws weight each official cell equally."""
    people = np.unique(subject)
    weights = np.array([len(np.unique(cell[subject == p])) for p in people])
    totals = np.zeros((2, len(stock), len(people)))
    for i, person in enumerate(people):
        for key in np.unique(cell[subject == person]):
            mask = cell == key
            for arm, decisions in enumerate((stock, trial)):
                totals[arm, :, i] += [100 * balanced_accuracy(y[mask], d[mask]) for d in decisions]
    baseline, score_trial = totals.sum((1, 2)) / (len(stock) * weights.sum())
    delta = totals[1] - totals[0]
    rng = np.random.default_rng(19)
    pi = rng.integers(len(people), size=(10000, len(people)))
    oi = rng.integers(len(stock), size=(len(pi), len(stock)))
    draws = delta[oi[:, :, None], pi[:, None, :]].sum((1, 2)) / (weights[pi].sum(1) * len(stock))
    return dict(stock_pct=baseline, trial_pct=score_trial, delta_pp=score_trial - baseline,
                i80=np.quantile(draws, [.1, .9]).tolist(), i95=np.quantile(draws, [.025, .975]).tolist(),
                by_order=(delta.sum(1) / weights.sum()).tolist(),
                by_person=dict(zip(people, (delta.mean(0) / weights).tolist())),
                fixed=int(((stock != y) & (trial == y)).sum()),
                broken=int(((stock == y) & (trial != y)).sum()), n_cells=int(weights.sum()))


def verify_stock(order, value):
    receipt = json.loads((ROOT / f"runs/niv/A_{order}_score.json").read_text())
    np.testing.assert_allclose(value, receipt["absolute_balanced_accuracy_pct"][STOCK], rtol=0, atol=1e-10)
