"""Plot signed cap250 factorial effects on the actual balanced accuracy metric."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot import BLUE, GOLD, INK, MUTED, save


RESULT_SHA = "f6aa0b53a02a249ac9cb85c79cb8ad613622039f1d2e6d7828d1baf9a509b641"


def draw(receipt, out):
    raw = receipt.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RESULT_SHA:
        raise ValueError("the score receipt differs from the signed factorial")
    data = json.loads(raw)
    contrasts = [
        ("Directional alignment", data["main"]["directional"]),
        ("Cognitive waveform odds", data["main"]["cognitive"]),
        ("Both changes", data["main"]["combined"]),
        ("Factorial interaction", data["interaction"]),
    ]
    fig, (total, order) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, layout="constrained")
    fig.get_layout_engine().set(rect=(.02, .12, .96, .86))
    fig.suptitle("The cap250 combination varied across deliveries", x=.025, ha="left",
                 fontsize=20, weight="bold", color=INK)
    total.set_title("Overall effect with paired 80% interval", loc="left", fontsize=10, color=MUTED)
    order.set_title("Effect by delivery order", loc="left", fontsize=10, color=MUTED)
    for i, (_, row) in enumerate(contrasts):
        pp = row["pp"]
        low, high = row["paired80"]
        total.errorbar(pp, i, xerr=[[pp-low], [high-pp]], color=INK, fmt="o", ms=6, capsize=3)
        for label, color, offset in (("2411", BLUE, -.105), ("3411", GOLD, .105)):
            order.scatter(row["order_pp"][label], i+offset, color=color,
                          label=f"Order {label}" if i == 0 else None)
    for ax in (total, order):
        ax.set_xlim(-1.05, 1.05)
        ax.set_xticks((-1., -.5, 0., .5, 1.))
        ax.set_ylim(len(contrasts)-.5, -.65)
        ax.axvline(0, lw=1, ls="--", color=MUTED)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e1e7ee")
    total.set_yticks(range(len(contrasts)), [label for label, _ in contrasts])
    order.legend(loc="upper left", frameon=False, ncol=2)
    fig.supxlabel("Change in balanced accuracy cell mean (percentage points)", fontsize=10, y=.077)
    fig.text(.025, .026, "Full system on Scherer EEG · 9 reused participants · 2 visits · 2 new delivery orders",
             color=MUTED, fontsize=9)
    out.mkdir(parents=True, exist_ok=True)
    save(fig, out, "cap250-alignment-cognitive-factorial")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    draw(options.receipt, options.output)
