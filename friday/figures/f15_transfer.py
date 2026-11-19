"""Compare F15's verified paired effects on PROTEUS and Scherer."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot import BLUE, GOLD, INK, MUTED, save


HELD_SHA = "eb06cd1fa8c52540706045850a4a76fa74113f7febe668e5229dc1a12984842e"
EXTERNAL_SHA = "604eb04ca1cd5536c96ef73d84c138d799d4fd5defd1a4cacf75559d3606312e"


def load(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"unexpected score receipt {path.name}")
    return json.loads(raw)


def draw(held_path, external_path, output):
    held = load(held_path, HELD_SHA)
    external = load(external_path, EXTERNAL_SHA)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10.5, 3.9), layout="constrained")
    fig.get_layout_engine().set(rect=(.025, .11, .96, .85))
    fig.suptitle("Feature dropout transfers more clearly against scaling",
                 x=.025, ha="left", fontsize=18, weight="bold", color=INK)
    for ax, control in zip(axes, ("scaled", "original"), strict=True):
        for seed, offset, color, marker in ((0, -.12, BLUE, "o"), (1, .12, GOLD, "D")):
            internal = held["seeds"][str(seed)]["paired"][f"dropout_vs_{control}_on"]
            transfer = external["contrasts"][f"s{seed}_bounded_dropout_minus_{control}"]
            for row, (value, bounds) in enumerate(((internal["mean"], internal["interval"]),
                                                   (transfer["pp"], transfer["paired80"]))):
                lo, hi = bounds
                ax.errorbar(value, row + offset, xerr=[[value - lo], [hi - value]],
                            color=color, fmt=marker, ms=6, capsize=3, lw=1.6,
                            label=f"Seed {seed}" if row == 0 else None)
        ax.set_title("Dropout vs fixed scaling" if control == "scaled" else "Dropout vs original training",
                     loc="left", fontsize=12, color=INK, pad=9, weight="bold")
        ax.axvline(0, lw=1, ls="--", color=MUTED)
        ax.set(xlim=(-1.05, 1.85), ylim=(1.65, -.7), xticks=[-1, 0, 1],
               xlabel="Balanced accuracy change (pp)")
        ax.set_yticks([0, 1], ["PROTEUS\n10 held-out people", "Scherer\n9 external people"])
        ax.grid(axis="x", color="#e1e7ee")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="upper left", frameon=False, ncol=2, fontsize=9)
    fig.text(.025, .035, "Full features and bounded normalization at inference. Bars show paired 80% intervals.",
             fontsize=9.2, color=MUTED)
    output.mkdir(parents=True, exist_ok=True)
    save(fig, output, "held-s1c-dropout-transfer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("held_receipt", type=Path)
    parser.add_argument("external_receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(args.held_receipt, args.external_receipt, args.output)
