"""Show which participants and tasks contributed to the signed F-C20 differences."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from native_adaptation import locked
from plot import INK, MUTED, save


SOURCE = "1100016fe85b769282b01d7df6bb374e5590f1707ca8375e2aa1e84ba7b4ae30"
ROWS = [(delivery, arm) for delivery in ("within0", "full0") for arm in ("hard", "jitter")]
TASKS = ("Motor\nimagery", "Mental\nsubtraction", "Word\ngeneration")


def matrix(ax, data, names, xlabel, precision, fig):
    extent = max(1., float(np.abs(data).max()))
    art = ax.imshow(data, cmap="RdBu", norm=TwoSlopeNorm(0, -extent, extent),
                    interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(names)), names)
    ax.tick_params(length=0)
    for row, column in np.ndindex(data.shape):
        value = data[row, column]
        label = "0" if abs(value) < .5 * 10 ** (-precision) else f"{value:+.{precision}f}"
        ax.text(column, row, label, ha="center", va="center", fontsize=9,
                color="white" if abs(value) > .62 * extent else INK)
    bar = fig.colorbar(art, ax=ax, location="bottom", shrink=.82, pad=.08, aspect=30)
    bar.set_label(xlabel, fontsize=9)
    ax.spines[:].set_visible(False)


def draw(receipts, out):
    record = locked(receipts / "held-outer-raw-replacement-score.json", SOURCE)["orders"]
    people = sorted(record["within0"]["people_pct"])
    per_person = []
    per_task = []
    for delivery, arm in ROWS:
        scope = record[delivery]
        assert set(people) == set(scope["people_pct"])
        per_person.append([scope["people_pct"][name][arm] - scope["people_pct"][name]["stock"]
                           for name in people])
        per_task.append([scope["class_changes"][str(task)][arm]["net"] for task in range(3)])
    fig, (participants, tasks) = plt.subplots(figsize=(12.6, 4.3), ncols=2, sharey=True,
                                              width_ratios=(2.7, 1), layout="constrained")
    fig.get_layout_engine().set(rect=(.015, .08, .975, .82))
    fig.suptitle("Raw replacement gains varied by participant and task", x=.027, ha="left",
                 weight="bold", fontsize=18, color=INK)
    participants.set_title("Accuracy difference by participant", loc="left", fontsize=11, color=INK, pad=10)
    tasks.set_title("Net correct decisions by task", loc="left", fontsize=11, color=INK, pad=10)
    matrix(participants, np.array(per_person), [name.rsplit("/", 1)[1] for name in people],
           "Balanced accuracy change (percentage points)", 2, fig)
    matrix(tasks, np.array(per_task), TASKS, "Fixes minus regressions (windows)", 0, fig)
    participants.set_yticks(range(len(ROWS)), [f"{order} · {arm}" for order, arm in ROWS])
    fig.text(.027, .020, "Full-system dev · each arm compared with exact stock in its replay · same 10 reused held-new participants",
             fontsize=9, color=MUTED)
    out.mkdir(parents=True, exist_ok=True)
    save(fig, out, "held-outer-raw-replacement-groups")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    draw(options.receipts, options.output)
