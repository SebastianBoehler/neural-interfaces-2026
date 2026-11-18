"""Plot the authenticated F14 paired training and inference contrasts."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot import BLUE, GOLD, INK, MUTED, save


RESULT_SHA = "82fe550621a8eb72a0762c507fb81ee1a9d4d3cc65983a3e8bbdf2d923aed420"
CONTRASTS = (
    ("original_bounded_eval", "Bounded inference\nOriginal training"),
    ("bounded_training", "Add bounding during training\nSame bounded inference"),
    ("causal_training_ordinary_eval", "Causal training\nSame ordinary inference"),
    ("whole_decoder", "Bounded training + inference\nVersus original S1C"),
)


def draw(receipt, out):
    raw = receipt.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RESULT_SHA:
        raise ValueError("F14 receipt differs from the verified source")
    data = json.loads(raw)

    fig, ax = plt.subplots(figsize=(9.6, 4.75), layout="constrained")
    fig.get_layout_engine().set(rect=(.02, .085, .96, .91))
    fig.suptitle("Original training gave the strongest waveform scores",
                 x=.025, ha="left", fontsize=19, weight="bold", color=INK)
    ax.set_title("Ten held-out PROTEUS participants · eight delivery orders",
                 loc="left", fontsize=10, color=MUTED, pad=10)
    for seed, color, marker, offset in (("0", BLUE, "o", -.12), ("1", GOLD, "D", .12)):
        results = data["seeds"][seed]["paired"]
        for i, (key, _) in enumerate(CONTRASTS):
            mean = results[key]["mean"]
            lo, hi = results[key]["interval"]
            ax.errorbar(mean, i + offset, xerr=[[mean - lo], [hi - mean]],
                        color=color, fmt=marker, ms=6, capsize=3, lw=1.65,
                        label=f"Seed {seed}" if i == 0 else None)
    ax.axvline(0, lw=1, ls="--", color=MUTED)
    ax.set_yticks(range(len(CONTRASTS)), [label for _, label in CONTRASTS])
    ax.set_ylim(len(CONTRASTS) - .45, -.55)
    ax.set(xlim=(-5, 2.2), xlabel="Paired change in standalone balanced accuracy (percentage points)")
    ax.grid(axis="x", color="#e1e7ee")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.text(.025, .029, "Bars show paired 80% intervals. Original training with bounded inference scored 59.25% and 60.26%.",
             fontsize=9.2, color=MUTED)
    out.mkdir(parents=True, exist_ok=True)
    save(fig, out, "held-s1c-training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(args.receipt, args.output)
