"""Plot signed native CSP arbitration and returning Eye contrasts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

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


def plot_multiresolution_stack(receipts, out):
    record = locked(receipts / "held-multiresolution-stack-score.json",
                    "05fc761a1988e05ae56e7166d8b018456d84dee9a169358c88b9f636f708ee43")
    fig, (ax,) = layout("Multiresolution beat the matched conventional gates")
    rows = (("Multiresolution vs conventional", "multi_minus_single"),
            ("Multiresolution vs hard-label pupil", "multi_minus_hard"),
            ("Multiresolution vs stock", "multi_minus_stock"))
    for name, color, offset in (("within0", BLUE, -.13), ("full0", GOLD, .13)):
        for row, (_, key) in enumerate(rows):
            interval(ax, record["orders"][name]["contrasts"][key], row + offset,
                     color, name if row == 0 else None)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.set(xlim=(-.28, .80), ylim=(2.56, -.55))
    ax.set_title("Complete cap250 consumer · paired 80% intervals", loc="left", fontsize=10, color=MUTED)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.text(.028, .024, "Full-system dev · same 10 held-new participants · one replay per configuration · stock retained",
             fontsize=9, color=MUTED)
    save(fig, out, "held-multiresolution-stack")


def plot_external_raw(receipts, out):
    record = locked(receipts / "external-outer-raw-same-input-score.json",
                    "6f3b1379eab3ae567e77601ca6744a34637b7f4e66024f49b8025347a226f211")
    fig, (ax,) = layout("The raw replacement had a near-zero external effect")
    effect, people = record["contrast"], record["people_pp"]
    interval(ax, effect, 0, BLUE)
    for row, (seed, p) in enumerate(sorted(effect["orders_pp"].items()), 1):
        ax.scatter(p, row, color=GOLD, s=40)
    ax.scatter(list(people.values()), 3 + np.linspace(-.12, .12, len(people)), color=MUTED, s=30)
    ax.set_yticks(range(4), ["Overall · paired 80%", "Delivery 8411", "Delivery 9411", "Individual participants"])
    ax.set(xlim=(-.91, .73), ylim=(3.59, -.57))
    ax.set_title("Hard-label raw replacement vs exact stock · Scherer", loc="left", fontsize=10, color=MUTED)
    fig.text(.028, .024, "Conditional full-system external · nine reused participants · both outer branches received identical native upstream state",
             fontsize=8.8, color=MUTED)
    save(fig, out, "external-outer-raw-same-input")


def plot_fbl_factor(receipts, out):
    record = locked(receipts / "held-fbl-doubling-factorial-score.json",
                    "cfd9584c29d3e9175228f6f92c633a09d63f91e20d88c6ae722c3e726a58f939")
    fig, (ax,) = layout("Doubling adapted FBLight improved both held-new replays")
    rows = (("Adapted FBLight double vs stock", "FBL2_minus_stock"),
            ("CC30 replacement vs stock", "SS_minus_stock"),
            ("Both changes vs stock", "JS_minus_stock"),
            ("Both changes vs FBLight double", "JS_minus_FBL2"))
    for name, color, offset in (("within0", BLUE, -.13), ("full0", GOLD, .13)):
        for row, (_, key) in enumerate(rows):
            interval(ax, record["orders"][name]["contrasts"][key], row + offset,
                     color, name if row == 0 else None)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.set(xlim=(-.49, 1.11), ylim=(3.59, -.55))
    ax.set_title("Complete updated cap250 replay · paired 80% intervals", loc="left", fontsize=10, color=MUTED)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.text(.028, .024, "Full-system dev · same 10 held-new participants · both comparators retain live refit and personalization",
             fontsize=9, color=MUTED)
    save(fig, out, "held-fbl-doubling-factorial")


def plot_returning_raw(receipts, out):
    record = locked(receipts / "returning-outer-raw-score.json",
                    "811d15ace58f38726d91692c7729ab26c177054868d700677203fc6a28a511d4")
    fig, (ax,) = layout("The raw replacement changed three returning decisions")
    rows = (("Equal official cells", "official_equal"), ("Returning 20/80 mix", "20_80"),
            ("P04", "P04"), ("P10", "P10"))
    for row, (_, key) in enumerate(rows):
        value = record["pairings"][key]
        interval(ax, value, row, BLUE if row < 2 else MUTED)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.set(xlim=(-.57, .62), ylim=(3.55, -.55))
    ax.set_title("Hard-label raw replacement vs exact native stock · paired 80% intervals", loc="left", fontsize=10, color=MUTED)
    fig.text(.028, .024, "Conditional full-system dev · four later visits of two participants · original gallery and production auxiliaries retained",
             fontsize=8.8, color=MUTED)
    save(fig, out, "returning-outer-raw")


def plot_joint_waveform(receipts, out):
    record = locked(receipts / "held-fbl-multiresolution-cross-score.json",
                    "309c3f6d368b4b1034e963233d54df61cb372b62b1b07f289cd95e1df26e63c0")
    fig, (ax,) = layout("The combined model improved on cap250 in both deliveries")
    rows = (("Exact cap250", "joint_minus_stock"), ("FBLight doubling alone", "joint_minus_FBL2"),
            ("Multiresolution pupil alone", "joint_minus_multi"),
            ("FBLight + conventional pupil", "joint_minus_single_joint"))
    for order, color, offset in (("within0", BLUE, -.11), ("full0", GOLD, .11)):
        for row, (_, key) in enumerate(rows):
            interval(ax, record["orders"][order]["contrasts"][key], row + offset,
                     color, order if row == 0 else None)
    ax.set_yticks(range(len(rows)), [name for name, _ in rows])
    ax.set(xlim=(-.32, 1.08), ylim=(3.55, -.55))
    ax.set_title("FBLight doubling + multiresolution pupil vs each reference · paired 80% intervals",
                 loc="left", fontsize=10, color=MUTED)
    ax.legend(loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(1, .78))
    fig.text(.028, .024, "Full-system dev · 10 held-new participants used in both delivery orders",
             fontsize=9, color=MUTED)
    save(fig, out, "held-fbl-multiresolution-cross")
    plot_joint_people(record, out)


def plot_joint_people(record, out):
    fig, (ax,) = layout("Three participants improved in both delivery orders", width=10)
    within, full = (record["orders"][name]["people_pct"] for name in ("within0", "full0"))
    assert within.keys() == full.keys()
    offsets = {"P02": (-7, 8), "P04": (7, -13), "P09": (7, -14), "P10": (-7, -14),
               "P03": (-8, 8), "P05": (7, -13), "P08": (7, 7), "P06": (-7, -14)}
    for person, votes in within.items():
        subject = person.rsplit("/", 1)[1]
        x = votes["joint"] - votes["FBL2"]
        y = full[person]["joint"] - full[person]["FBL2"]
        ax.scatter(x, y, s=29, color=BLUE if x > 0 and y > 0 else MUTED, zorder=3)
        dx, dy = offsets.get(subject, (6, 8))
        ax.annotate(subject, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="right" if dx < 0 else "left", fontsize=8.5, color=INK)
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=.85, zorder=1)
    ax.set(xlim=(-1.25, 1.26), ylim=(-.75, 1.24), ylabel="full0 gain (pp)")
    fig.supxlabel("within0 gain (pp)", fontsize=10, y=.076)
    ax.set_title("FBLight + multiresolution pupil vs FBLight doubling alone", loc="left",
                 fontsize=10, color=MUTED)
    fig.text(.028, .024, "Full-system dev · same ten held-new people · each point averages one person's official cells",
             fontsize=8.7, color=MUTED)
    save(fig, out, "held-fbl-multiresolution-people")


def plot_fbl_dose(receipts, out):
    record = locked(receipts / "held-fblight-dose-score.json",
                    "e8760cf01a09a6a856a73bc22a4096cb3bf5aa64096099a6659390d5306c1dfd")
    fig, (ax,) = layout("More FBLight weight gave different results by delivery")
    rows = (
        ("Live doubled vs cap250", "FBL2_minus_stock"),
        ("Frozen addition vs cap250", "frozen_minus_stock"),
        ("Live tripled vs cap250", "FBL3_minus_stock"),
        ("Live doubled vs frozen addition", "FBL2_minus_frozen"),
        ("Live tripled vs live doubled", "FBL3_minus_FBL2"),
    )
    for order, color, offset in (("within0", BLUE, -.11), ("full0", GOLD, .11)):
        for row, (_, key) in enumerate(rows):
            interval(ax, record["orders"][order]["contrasts"][key], row + offset,
                     color, order if row == 0 else None)
    ax.set_yticks(range(len(rows)), [name for name, _ in rows])
    ax.set(xlim=(-.84, 1.18), ylim=(4.55, -.55))
    ax.set_title("Exact cap250 and three FBLight weighting rules · paired 80% intervals",
                 loc="left", fontsize=10, color=MUTED)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.text(.028, .024, "Full-system dev · same ten held-new participants in both deliveries",
             fontsize=9, color=MUTED)
    save(fig, out, "held-fblight-dose")


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
    plot_multiresolution_stack(options.receipts, options.output)
    plot_external_raw(options.receipts, options.output)
    plot_fbl_factor(options.receipts, options.output)
    plot_returning_raw(options.receipts, options.output)
    plot_joint_waveform(options.receipts, options.output)
    plot_fbl_dose(options.receipts, options.output)
