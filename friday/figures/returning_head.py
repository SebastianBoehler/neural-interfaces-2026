"""Plot the two-person conditional waveform-head comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot import BLUE, GOLD, INK, MUTED, save


SOURCE = "8224cf7a157d39bbbe3b132d04a2c6b1dd18b94a781624a277de363481446962"


def draw(receipt, out):
    content = receipt.read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE:
        raise ValueError("the returning result differs from its signed receipt")
    data = json.loads(content)
    fig, ax = plt.subplots(figsize=(9.3, 5.1), layout="constrained")
    fig.get_layout_engine().set(rect=(.02, .092, .96, .88))
    fig.suptitle("The first-visit waveform head helped P04 and hurt P10", x=.025,
                 ha="left", fontsize=17.5, weight="bold", color=INK)
    ax.set_title("Conditional cap250 replay · 762 windows · one delivery order",
                 loc="left", fontsize=10, color=MUTED)
    visits = ("P04|2", "P04|3", "P10|2", "P10|3")
    for name, label, color, marker, offset in (
        ("own_vs_native", "Own vs native", BLUE, "o", -.12),
        ("own_vs_population", "Own vs population", GOLD, "D", .12),
    ):
        effect = data["contrasts"][name]
        points = [effect["by_visit_pp"][visit] for visit in visits]
        points.extend((effect["pp_all_cells"], effect["pp_weighted"]))
        ax.scatter(points, [i+offset for i in range(6)], s=45, color=color, marker=marker, label=label)
    ax.set_yticks(range(6), [visit.replace("|", " visit ") for visit in visits]
                  + ["All cells", "20% P04 / 80% P10"])
    ax.set(xlim=(-2.45, 3.55), ylim=(5.6, -.68),
           xlabel="Change in balanced accuracy cell mean (percentage points)")
    ax.axvline(0, color=MUTED, ls="--", lw=1)
    ax.axhline(3.5, color="#ccd7e7", lw=1)
    ax.grid(axis="x", color="#e1e7ee")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.text(.025, .026, "Two known participants. The replay retains production auxiliary and calibration exposure.",
             fontsize=9, color=MUTED)
    out.mkdir(exist_ok=True, parents=True)
    save(fig, out, "returning-first-visit-waveform-head")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    draw(options.receipt, options.output)
