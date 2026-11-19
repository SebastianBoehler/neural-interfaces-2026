"""Plot signed native CSP arbitration and returning Eye contrasts."""

import argparse
import hashlib
import json
from pathlib import Path

from adaptation import interval, layout
from plot import BLUE, GOLD, INK, MUTED, save


def locked(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{path.name} differs from the signed result")
    return json.loads(raw)


def plot_arbitration(receipts, out):
    record = locked(receipts / "held-csp-disagreement-score.json",
                    "675c23bf75c75b25a793e8b96abf6bfce37673509dd1d7a82dc64baaeddc7707")
    fig, (ax,) = layout("The CSP rule recovered little of the available accuracy")
    rows = (
        ("CSP odds rule", "gate_minus_stock"),
        ("Prior-window control", "sham_minus_stock"),
        ("Equal log blend", "blend_minus_stock"),
        ("Label-aware choice · eligible", "oracle_eligible_minus_stock"),
        ("Label-aware choice · all windows", "oracle_unrestricted_minus_stock"),
    )
    ax.axhspan(2.55, 4.48, color="#eef2f6", zorder=0)
    for name, color, offset in (("within0", BLUE, -.11), ("full0", GOLD, .11)):
        for row, (_, key) in enumerate(rows):
            interval(ax, record["orders"][name]["contrasts"][key], row + offset,
                     color, name if row == 0 else None)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.set(xlim=(-1., 3.52), ylim=(4.5, -.5))
    ax.set_title("Against exact stock in each replay · paired 80% intervals", loc="left", fontsize=10, color=MUTED)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.text(.028, .024, "Full-system dev · same 10 held participants · shaded rows use the correct labels to choose a branch",
             fontsize=9, color=MUTED)
    save(fig, out, "held-csp-disagreement-ceiling")


def plot_eye(receipts, out):
    record = locked(receipts / "returning-source-eye-score.json",
                    "63a44211d7e8b45780970c8e7e1b79f9eddcb4f991b01c05149a09bcb528168c")
    fig, (ax,) = layout("Replacing only the Eye fit lowered returning accuracy")
    rows = (("Equal official cells", "official_equal"), ("Returning 20/80 mix", "20_80"), ("P04", "P04"), ("P10", "P10"))
    for row, (_, key) in enumerate(rows):
        value = record["pairings"][key]
        interval(ax, value, row, BLUE if row == 0 else MUTED)
        ax.annotate(f"{value['pp']:+.2f}", (value["paired80"][1], row), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.set(xlim=(-5.05, .25), ylim=(3.5, -.5))
    ax.set_title("Source-fold Eye vs production Eye · paired 80% intervals", loc="left", fontsize=10, color=MUTED)
    fig.text(.028, .024, "Conditional full-system dev · two reused participants · original Eye and gallery calibration retained",
             fontsize=9, color=MUTED)
    save(fig, out, "returning-source-eye")


def plot_eye_calibration(receipts, out):
    record = locked(receipts / "returning-eye-calibration-score.json",
                    "54c7b7dfdf3488dce061fdbd526e6237c0e04d4f689e44770789f5c420abcde0")
    fig, (ax,) = layout("First-visit Eye calibration helped P04 and hurt P10")
    rows = (("Equal official cells", "official_equal"), ("Returning 20/80 mix", "20_80"),
            ("P04", "P04"), ("P10", "P10"))
    for row, (_, key) in enumerate(rows):
        value = record["contrasts"]["calibration_vs_source"][key]
        interval(ax, value, row, BLUE if row < 2 else MUTED)
        ax.annotate(f"{value['pp']:+.2f}", (value["paired80"][1], row), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.set(xlim=(-.54, .94), ylim=(3.55, -.55))
    ax.set_title("Calibrated vs identical source-only Eye · paired 80% intervals", loc="left", fontsize=10, color=MUTED)
    fig.text(.028, .024, "Conditional full-system dev · four held visits from two reused participants · original gallery calibration",
             fontsize=9, color=MUTED)
    save(fig, out, "returning-eye-calibration")


def plot_outer_raw(receipts, out):
    record = locked(receipts / "held-outer-raw-replacement-score.json",
                    "1100016fe85b769282b01d7df6bb374e5590f1707ca8375e2aa1e84ba7b4ae30")
    fig, (ax,) = layout("The raw replacement gained points in both replays")
    rows = (("Hard-label pupil vs stock", "hard_minus_stock"),
            ("Jitter pupil vs stock", "jitter_minus_stock"),
            ("Jitter pupil vs hard-label pupil", "jitter_minus_hard"))
    for name, color, offset in (("within0", BLUE, -.13), ("full0", GOLD, .13)):
        for row, (_, key) in enumerate(rows):
            value = record["orders"][name]["contrasts"][key]
            interval(ax, value, row + offset, color, name if row == 0 else None)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.set(xlim=(-.31, .74), ylim=(2.55, -.55))
    ax.set_title("Complete cap250 consumer · paired 80% intervals", loc="left", fontsize=10, color=MUTED)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.text(.028, .024, "Full-system dev · same 10 held-new participants · one replay per delivery order · stock retained",
             fontsize=9, color=MUTED)
    save(fig, out, "held-outer-raw-replacement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    plot_arbitration(options.receipts, options.output)
    plot_eye(options.receipts, options.output)
    plot_eye_calibration(options.receipts, options.output)
    plot_outer_raw(options.receipts, options.output)
