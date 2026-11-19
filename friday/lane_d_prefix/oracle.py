"""First Lane D decision: fit the preregistered whole-session Gaussian and seal labels."""

import json
from pathlib import Path
import sys

import numpy as np

from balance import terminal_probability
from banks import DATA, ORDERS, ROOT, digest, metadata, score, stock_bank, verify_manifest, verify_stock

BANDS = ("theta", "alpha", "beta1", "beta2", "broad")


def log_power():
    def band(name):
        cov = np.load(ROOT / f"data/feature_cache/cov_{name}.npy", mmap_mode="r")
        return np.log(np.maximum(cov.diagonal(axis1=1, axis2=2), 1e-12))

    return np.concatenate([band(name) for name in BANDS], axis=1)


def gaussian(x, p, source_variance, strength=0.25):
    if strength == 0:
        return p
    x = x - x.mean(0)
    responsibility = p / p.sum(1, keepdims=True)
    mass = responsibility.sum(0)
    means = (responsibility.T @ x) / (mass[:, None] + 25)
    error = x[:, None] - means
    within = np.einsum("nk,nkd,nkd->d", responsibility, error, error) / len(x)
    variance = np.maximum(0.5 * within + 0.5 * source_variance, 1e-6)
    # The x² term in every class log likelihood is identical and cancels.
    logits = (x @ (means / variance).T - 0.5 * (means**2 / variance).sum(1)) / 41
    logits -= logits.mean(1, keepdims=True)
    return np.log(np.maximum(p, 1e-300)) + strength * np.clip(logits, -2, 2)


def derive(output):
    subject, visit, _ = metadata()
    features = log_power()
    people = np.unique(subject)
    variances = np.stack([features[subject == person].var(0) for person in people])
    bank = {}
    for order in ORDERS:
        delivery, stock, terminal, _ = stock_bank(order, len(subject))
        terminal = terminal_probability(terminal)
        predictions = stock.copy()
        for session in np.unique(visit):
            rows = delivery[visit[delivery] == session]
            training = variances[people != subject[rows[0]]].mean(0)
            x, p = features[rows], terminal[rows]
            np.testing.assert_array_equal(gaussian(x, p, training, 0), p)
            values = gaussian(x, p, training)
            permuted = gaussian(x[::-1], p[::-1], training)[::-1]
            np.testing.assert_allclose(values, permuted, rtol=1e-12, atol=1e-12)
            predictions[rows] = values.argmax(1)
        bank[order + "_stock"], bank[order + "_oracle"] = stock, predictions
    with output.open("xb") as handle:
        np.savez_compressed(handle, **bank)


def evaluate(predictions, output):
    seal = digest(predictions)
    with np.load(DATA, allow_pickle=False) as data:
        y = data["y"]
    subject, _, cell = metadata()
    result = {"blind_sha256": seal}
    with np.load(predictions, allow_pickle=False) as bank:
        for order in ORDERS:
            result[order] = score(y, bank[order + "_stock"][None], bank[order + "_oracle"][None], subject, cell)
            verify_stock(order, result[order]["stock_pct"])
    pp = result["full0"]["delta_pp"]
    result["decision"] = "stop_Da_Db" if pp < 0.50 else "scale_Da_Db" if pp >= 1.50 else "undecided"
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    verify_manifest()
    blind = directory / "oracle_blind.npz"
    derive(blind)
    evaluate(blind, directory / "oracle.json")
