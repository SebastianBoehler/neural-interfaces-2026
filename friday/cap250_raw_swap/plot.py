"""Plot paired person-level effects from the authenticated F-C13 result."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STYLE = {
    "seed0": ("#116C65", "o", "Seed 0 · primary"),
    "seed1": ("#7956B2", "D", "Seed 1 · replication"),
}


def plot(result, output):
    contrasts = result["contrasts"]
    people = sorted(contrasts["seed0"]["people_pp"])
    if any(set(data["people_pp"]) != set(people) for data in contrasts.values()):
        raise ValueError("the seed comparisons must cover the same people")
    fig, ax = plt.subplots(figsize=(8.7, 5.8), facecolor="white")
    fig.subplots_adjust(left=.15, right=.95, top=.74, bottom=.18)
    fig.text(.15, .93, "One raw waveform decoder swap in cap250", fontsize=18,
             fontweight="bold", color="#16202A")
    fig.text(.15, .875, "Full system on Scherer EEG · 9 people, 2 visits and 2 delivery orders",
             fontsize=10, color="#42556A")
    limits = []
    for i, (arm, data) in enumerate(contrasts.items()):
        color, marker, label = STYLE[arm]
        offset = (.14 if i == 0 else -.14) if len(contrasts) > 1 else 0
        y = [len(people) - j + offset for j in range(len(people))]
        scores = [data["people_pp"][p] for p in people]
        ax.scatter(scores, y, marker=marker, s=45, color=color, label=label, zorder=4)
        low, high = data["paired80"]
        ax.hlines(offset, low, high, color=color, linewidth=2.3, zorder=3)
        ax.scatter([data["pp"]], [offset], marker=marker, s=58, color=color, zorder=4)
        limits.extend([*scores, low, high, data["pp"]])
    bound = max(.25, *(abs(value) for value in limits)) * 1.15
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-.6, len(people) + .55)
    ax.set_yticks([*range(len(people), 0, -1), 0],
                  [p.rsplit("/", 1)[-1] for p in people] + ["Overall"])
    ax.axvline(0, color="#637385", linewidth=1.2)
    ax.axhline(.5, color="#BCC8D1", linewidth=.8)
    ax.grid(axis="y", color="#E7EDF1", linewidth=.65)
    ax.tick_params(axis="both", size=0, pad=7, labelcolor="#263545")
    ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
    ax.set_xlabel("Change in balanced accuracy cell mean vs. cap250 (percentage points)",
                  color="#263545", labelpad=9)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncols=len(contrasts),
              frameon=False, borderaxespad=0)
    fig.text(.15, .045, "Overall lines show paired 80% intervals, resampling people, visits and orders.",
             fontsize=9, color="#42556A")
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.result.read_text()), args.output)
