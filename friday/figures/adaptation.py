"""Visualize the authenticated cap250 replay and stage factorial."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot import BLUE, GOLD, INK, MUTED, save


RECEIPTS = {
    "within0": ("held-within0-alignment-score.json", "e4cd0a0e95a2202f872d4287db23dc977bee9f2faf6305f86dd75ce10609d297"),
    "full0": ("held-full0-alignment-score.json", "41a10bc6725cc572d80b06e416696d7aa1bc84fb6e3fb746e96527c9d6230033"),
    "stages": ("held-within0-adaptation-score.json", "a40f95125fa4b7df56cfea8176f240704ca0961995bb4b5f5ab0afa282d75d14"),
}


def contrasts(receipt_dir, name):
    filename, digest = RECEIPTS[name]
    raw = (receipt_dir / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{name} score differs from the authenticated receipt")
    return json.loads(raw)["contrasts"]


def interval(ax, value, y, color, label=None):
    mid = value["pp"]
    low, high = value["paired80"]
    ax.errorbar(mid, y, xerr=[[mid - low], [high - mid]], fmt="o", ms=5.5,
                capsize=3, lw=1.6, color=color, label=label)


def layout(title, width=10, columns=1):
    fig, panels = plt.subplots(1, columns, figsize=(width, 4.5), layout="constrained",
                              squeeze=False, sharey=True)
    fig.get_layout_engine().set(rect=(.025, .12, .95, .79))
    fig.suptitle(title, x=.028, ha="left", fontsize=18, weight="bold", color=INK)
    for ax in panels[0]:
        ax.axvline(0, color=MUTED, lw=.9, ls="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e1e7ee")
    fig.supxlabel("Change in balanced accuracy cell mean (percentage points)", fontsize=10, y=.076)
    return fig, panels[0]


def plot_alignment(receipt_dir, out):
    fig, (ax,) = layout("Cap250 kept a positive lead after alignment correction")
    comparisons = (
        ("Historical alignment\nCap250 vs P0r", "historical_cap250_minus_incumbent"),
        ("Corrected alignment\nCap250 vs P0r", "corrected_cap250_minus_incumbent"),
        ("Change in Cap250 lead\nCorrected vs historical", "corrected_minus_historical_gain"),
    )
    for name, color, offset in (("within0", BLUE, -.12), ("full0", GOLD, .12)):
        data = contrasts(receipt_dir, name)
        for row, (_, key) in enumerate(comparisons):
            interval(ax, data[key], row + offset, color, name if row == 0 else None)
    ax.set_yticks(range(3), [label for label, _ in comparisons])
    ax.set(ylim=(2.47, -.55), xlim=(-1, 3.85))
    ax.legend(loc="lower right", frameon=False)
    fig.text(.028, .024, "Full-system dev · 10 reused held participants · paired 80% intervals · replay configurations differ beyond delivery",
             fontsize=9, color=MUTED)
    save(fig, out, "held-alignment-two-replays")


def plot_stages(receipt_dir, out):
    fig, (score, people) = layout("Fine-tuning and CSP each helped the corrected replay", width=11.4, columns=2)
    data = contrasts(receipt_dir, "stages")
    comparisons = (
        ("Fine-tuning\nwith CSP", "ft_given_csp"),
        ("CSP\nwith fine-tuning", "csp_given_ft"),
        ("Both stages\nvs neither", "stock_minus_both"),
        ("Fine-tuning × CSP\ninteraction", "interaction"),
    )
    score.set_title("Across participants · paired 80% interval", loc="left", color=MUTED, fontsize=10)
    people.set_title("Individual participant changes", loc="left", color=MUTED, fontsize=10)
    for row, (_, key) in enumerate(comparisons):
        value = data[key]
        interval(score, value, row, BLUE)
        score.annotate(f"{value['pp']:+.2f}", (value["paired80"][1], row), xytext=(7, 0),
                       textcoords="offset points", va="center", fontsize=9, color=INK)
        per_person = {name.rsplit("/", 1)[1]: pp for name, pp in value["per_person_pp"].items()}
        others = {name: pp for name, pp in per_person.items() if name not in {"P04", "P10"}}
        people.scatter(list(others.values()), row + np.linspace(-.12, .12, len(others)), s=20,
                       color=MUTED, alpha=.55, label="Other participants" if row == 0 else None)
        for name, color, marker, offset in (("P04", GOLD, "s", -.13), ("P10", BLUE, "D", .13)):
            people.scatter(per_person[name], row + offset, s=31, color=color, marker=marker,
                           label=name if row == 0 else None)
    score.set_yticks(range(4), [label for label, _ in comparisons])
    score.set(ylim=(3.62, -.5), xlim=(-.95, 3.05))
    people.legend(loc="lower right", ncol=3, frameon=False, fontsize=8)
    fig.text(.028, .024, "Full-system dev · corrected within0 only · 10 reused held participants · common spatial patterns (CSP)",
             fontsize=9, color=MUTED)
    save(fig, out, "held-within0-adaptation-stages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt_dir", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    plot_alignment(options.receipt_dir, options.output)
    plot_stages(options.receipt_dir, options.output)
