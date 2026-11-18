"""Plot authenticated S1C training and inference contrasts."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot import BLUE, GOLD, INK, MUTED, save


LAYOUTS = {
    "82fe550621a8eb72a0762c507fb81ee1a9d4d3cc65983a3e8bbdf2d923aed420": dict(
        title="Original training gave the strongest waveform scores",
        stem="held-s1c-training", xlim=(-5, 2.2),
        caption="Original training with bounded inference scored 59.25% and 60.26%.",
        contrasts=(
            ("original_bounded_eval", "Bounded inference\nOriginal training"),
            ("bounded_training", "Add bounding during training\nSame bounded inference"),
            ("causal_training_ordinary_eval", "Causal training\nSame ordinary inference"),
            ("whole_decoder", "Bounded training + inference\nVersus original S1C"),
        )),
    "eb06cd1fa8c52540706045850a4a76fa74113f7febe668e5229dc1a12984842e": dict(
        title="Masking fixed features helped when retained at inference",
        stem="held-s1c-feature-dropout", xlim=(-1.6, 1.72),
        caption="Masking during training with all features at inference scored 59.94% and 60.60%.",
        contrasts=(
            ("dropout_vs_scaled_on", "Masking vs 75% scaling in training\nFull features at inference"),
            ("dropout_vs_original_on", "Masking vs original training\nFull features at inference"),
            ("dropout_vs_zero_zero", "Masking vs zero features in training\nZero fixed features at inference"),
        )),
}


def draw(receipt, out):
    raw = receipt.read_bytes()
    key = hashlib.sha256(raw).hexdigest()
    if key not in LAYOUTS:
        raise ValueError("receipt differs from the verified sources")
    data = json.loads(raw)
    layout = LAYOUTS[key]
    contrasts = layout["contrasts"]

    fig, ax = plt.subplots(figsize=(9.6, 4.75), layout="constrained")
    fig.get_layout_engine().set(rect=(.02, .085, .96, .91))
    fig.suptitle(layout["title"], x=.025, ha="left", fontsize=19, weight="bold", color=INK)
    ax.set_title("Ten held-out PROTEUS participants · eight delivery orders",
                 loc="left", fontsize=10, color=MUTED, pad=10)
    for seed, color, marker, offset in (("0", BLUE, "o", -.12), ("1", GOLD, "D", .12)):
        results = data["seeds"][seed]["paired"]
        for i, (key, _) in enumerate(contrasts):
            mean = results[key]["mean"]
            lo, hi = results[key]["interval"]
            ax.errorbar(mean, i + offset, xerr=[[mean - lo], [hi - mean]],
                        color=color, fmt=marker, ms=6, capsize=3, lw=1.65,
                        label=f"Seed {seed}" if i == 0 else None)
    ax.axvline(0, lw=1, ls="--", color=MUTED)
    ax.set_yticks(range(len(contrasts)), [label for _, label in contrasts])
    ax.set_ylim(len(contrasts) - .45, -.55)
    ax.set(xlim=layout["xlim"], xlabel="Paired change in standalone balanced accuracy (percentage points)")
    ax.grid(axis="x", color="#e1e7ee")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.text(.025, .029, f'Bars show paired 80% intervals. {layout["caption"]}', fontsize=9.2, color=MUTED)
    out.mkdir(parents=True, exist_ok=True)
    save(fig, out, layout["stem"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(args.receipt, args.output)
