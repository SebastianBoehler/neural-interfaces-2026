"""Draw the signed native returning F-C21r result and its paired visits."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot import BG, BLUE, GOLD, INK, MUTED, save


RECEIPT = "e1837c2f59a6f58333e4ed336e213b07e4a77c238e54d5ede50202324d935847"


def draw(receipt, output):
    raw = receipt.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RECEIPT:
        raise ValueError("returning score receipt differs from signed F-C21r")
    data = json.loads(raw)
    fig, (overall, visits) = plt.subplots(1, 2, figsize=(9.7, 4.45), layout="constrained", facecolor=BG)
    fig.get_layout_engine().set(rect=(.03, .13, .94, .77))
    fig.suptitle("Doubling FBLight gave no clear returning gain", x=.035, y=.96,
                 ha="left", color=INK, fontsize=19, weight="bold")
    overall.set_title("Returning people · paired 80% intervals", loc="left", color=INK, fontsize=10, weight="bold")
    labels = ("Equal official cells", "20/80 returning mix", "P04", "P10")
    for i, (key, color) in enumerate(zip(("official_equal", "20_80", "P04", "P10"), (INK, GOLD, BLUE, BLUE), strict=True)):
        r = data["pairings"][key]
        pp = r["pp"]
        lo, hi = r["paired80"]
        overall.errorbar(pp, i, xerr=[[pp - lo], [hi - pp]], fmt="o", color=color, lw=1.3, capsize=3, ms=5)
        overall.text(.56, i, f"{pp:+.3f}" if abs(pp) >= .0005 else "0.000", ha="right", va="center", fontsize=8.6, color=color)
    overall.set_yticks(range(4), labels)
    overall.set(xlim=(-.47, .60), xticks=(-.4, -.2, 0, .2, .4), ylim=(3.55, -.55))
    visits.set_title("Same visits · point changes", loc="left", color=INK, fontsize=10, weight="bold")
    visit_labels = []
    for i, who in enumerate(data["people"]):
        for j, (day, color) in enumerate((("2", BLUE), ("3", GOLD))):
            key = f"{who}|{day}"
            pp = np.diff(np.array(list(data["sessions"][key]["cells"].values())), axis=1).mean().item()
            row = 2 * i + j
            visit_labels.append(f"{who} · visit {day}")
            visits.hlines(row, min(0, pp), max(0, pp), color=color, linewidth=1.2)
            visits.scatter(pp, row, color=color, marker="o" if j == 0 else "D", s=28)
            visits.text(.73, row, f"{pp:+.3f}" if abs(pp) >= .0005 else "0.000", ha="right", va="center", fontsize=8.6, color=color)
    visits.set_yticks(range(4), visit_labels)
    visits.set(xlim=(-.60, .78), xticks=(-.5, -.25, 0, .25, .5), ylim=(3.55, -.55))
    for ax in (overall, visits):
        ax.set_facecolor(BG)
        ax.axvline(0, color=MUTED, linestyle="--", linewidth=.8, zorder=0)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(colors=INK, labelsize=8.6)
    fig.supxlabel("Change in balanced accuracy vs. native stock (percentage points)", y=.09, color=INK, fontsize=9)
    fig.text(.035, .026, "FULL-SYSTEM DEV · two reused returning people · four full0 visits · original gallery and production auxiliaries", color=MUTED, fontsize=8.2)
    output.mkdir(parents=True, exist_ok=True)
    save(fig, output, "f-c21r-fblight-returning")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    draw(a.receipt, a.output)
