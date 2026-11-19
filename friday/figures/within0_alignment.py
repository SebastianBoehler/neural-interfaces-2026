"""Plot the locked complete within0 alignment comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot import BLUE, GOLD, INK, MUTED, save


SOURCE = "e4cd0a0e95a2202f872d4287db23dc977bee9f2faf6305f86dd75ce10609d297"


def draw(receipt, out):
    raw = receipt.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE:
        raise ValueError("the within0 score differs from the pinned receipt")
    data = json.loads(raw)
    fig, ax = plt.subplots(figsize=(9.5, 4.1), layout="constrained")
    fig.get_layout_engine().set(rect=(.03, .10, .94, .86))
    fig.suptitle("Cap250 kept its within0 advantage after alignment correction",
                 ha="left", x=.032, fontsize=17, weight="bold", color=INK)
    ax.set_title("Complete held-new replay · 10 reused participants · one delivery order",
                 loc="left", fontsize=10, color=MUTED, pad=10)
    comparisons = (
        ("Historical alignment\nCap250 vs P0r", "historical_cap250_minus_incumbent", MUTED),
        ("Corrected alignment\nCap250 vs P0r", "corrected_cap250_minus_incumbent", BLUE),
        ("Change in Cap250 advantage\nCorrected vs historical", "corrected_minus_historical_gain", GOLD),
    )
    for row, (_, key, color) in enumerate(comparisons):
        result = data["contrasts"][key]
        mean = result["pp"]
        low, high = result["paired80"]
        ax.errorbar(mean, row, xerr=[[mean-low], [high-mean]], color=color,
                    fmt="o", ms=6, capsize=3, lw=1.7)
        ax.annotate(f"{mean:+.2f} pp", (high, row), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK)
    ax.set_yticks(range(3), [label for label, _, _ in comparisons])
    ax.set(xlim=(-1.05, 4.35), ylim=(2.55, -.55),
           xlabel="Change in balanced accuracy cell mean (percentage points)")
    ax.axvline(0, lw=1, ls="--", color=MUTED)
    ax.grid(axis="x", color="#e1e7ee")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.032, .025, "Paired 80% intervals resample people. Both arms share fresh Sinc and FBLight seed-0 states.",
             fontsize=9, color=MUTED)
    out.mkdir(parents=True, exist_ok=True)
    save(fig, out, "held-within0-alignment-advantage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    draw(options.receipt, options.output)
