"""Plot Friday's authenticated waveform outcomes and A100 profiles."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

INK = "#172940"
MUTED = "#52647a"
BLUE = "#1968a6"
GOLD = "#b36d19"
TEAL = "#087e78"
BG = "#f5f7fb"
STOCK = 76.10391516713594
SEEDS = (460, 1460)
MODELS = (
    ("prototype", "Prototype", "prototype"),
    ("prototype", "Centered context", "centered"),
    ("time", "Chronological", "chronological"),
    ("time", "Reversal-invariant", "symmetric"),
    ("time", "Coherent shifts", "coherent_time"),
    ("time", "Scattered shifts", "scattered_time"),
)
PEOPLE = [f"P{i:02}" for i in range(1, 11)]


def read(path):
    return json.loads(path.read_text())


def save(fig, out, stem):
    fig.savefig(out / (stem + ".png"), dpi=170, facecolor=BG)
    fig.savefig(out / (stem + ".pdf"), facecolor=BG)
    plt.close(fig)


def wave_plot(root, out):
    rows = []
    for folder, label, name in MODELS:
        for s in SEEDS:
            d = read(root / folder / f"{name}_{s}_result.json")
            if abs(d["absolute"]["stock"]["pp"] - STOCK) > 1e-8:
                raise ValueError("mismatched complete-stock score")
            rows.append((label, s, d["contrasts"]["full_genuine-stock"],
                         d["absolute"]["wave_genuine"]["pp"]))
    heat = np.array([[r[2]["people"][p] for p in PEOPLE] for r in rows])
    n = len(rows)
    fig = plt.figure(figsize=(12.5, 7.4), layout="constrained")
    fig.get_layout_engine().set(rect=(.005, .04, .99, .955))
    ax = fig.subplots(1, 3, width_ratios=[1.0, 2.05, .66], sharey=True)
    fig.suptitle("Single waveform decoders trail cap250 on held people", x=.022, ha="left",
                 fontsize=20, weight="bold")
    for a, title in zip(ax, ["Full stack vs stock", "Full-stack change by held person", "Whole decoder"]):
        a.set_title(title, loc="left", pad=11, fontsize=12.5, weight="bold")
    for y, (_, seed, contrast, whole) in enumerate(rows):
        m = contrast["pp"]
        lo, hi = contrast["paired80"]
        color = BLUE if seed == 460 else GOLD
        ax[0].errorbar(m, y, xerr=[[m-lo], [hi-m]], fmt="o" if seed == 460 else "D",
                       ms=5.5, lw=1.65, capsize=2.5, color=color, zorder=3)
        ax[2].scatter(whole, y, s=32, marker="o" if seed == 460 else "D", color=color, zorder=3)
    ax[0].axvline(0, color=MUTED, lw=.9, ls=(0, (4, 3)))
    img = ax[1].imshow(heat, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0, vmin=-2.5, vmax=2.5),
                       aspect="auto", interpolation="nearest")
    cb = fig.colorbar(img, ax=ax[1], location="bottom", shrink=.8, pad=.06, extend="both")
    cb.set_label("Per-person change (percentage points)", fontsize=9)
    ax[0].set_yticks(range(n), [f"{label}  ·  {seed}" for label, seed, _, _ in rows])
    ax[0].set_ylim(n-.5, -.5)
    ax[1].set_xticks(range(len(PEOPLE)), PEOPLE, rotation=45, ha="right")
    ax[0].set_xlabel("Change (pp), paired 80% interval")
    ax[2].set_xlabel("Balanced accuracy (%)")
    ax[2].set_xlim(32, 82)
    ax[2].axvline(100/3, color=MUTED, lw=.8, ls=":")
    ax[2].axvline(STOCK, color=MUTED, lw=1, ls="--")
    ax[0].grid(axis="x", alpha=.16)
    ax[2].grid(axis="x", alpha=.2)
    fig.text(.02, .01, "18 Nov 2026  ·  10 fit-excluded PROTEUS people  ·  Eight pinned orders  ·  Stock cap250 76.10%", fontsize=9, color=MUTED)
    save(fig, out, "held-waveform")


def runtime_plot(root, out):
    fig, (kernel, worker) = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    fig.get_layout_engine().set(rect=(.005, .065, .99, .925))
    fig.suptitle("A100 timing for two spectral kernel seeds", x=.02, ha="left",
                 fontsize=20, weight="bold")
    kernel.set_title("Isolated one-window kernel calls", loc="left", fontsize=12.5, weight="bold")
    worker.set_title("Complete paired cap250 sessions", loc="left", fontsize=12.5, weight="bold")
    methods = [("serial", "Serial", MUTED), ("batch", "Batched", BLUE), ("graph", "CUDA graph", TEAL)]
    for i, person in enumerate(("P04", "P10")):
        data = read(root / "profile" / f"{person}.json")
        for offset, (name, label, color) in zip((-.22, 0, .22), methods):
            values = data["times_ms"][name]
            kernel.plot(values, [i+offset]*len(values), "o", color=color, mfc="none",
                        ms=4, label=label if i == 0 else None)
    kernel.set_yticks([0, 1], ["P04 · 189 windows", "P10 · 192 windows"])
    kernel.set_ylim(1.55, -.55)
    kernel.set_xlabel("Milliseconds per window")
    kernel.legend(loc="lower right", fontsize=8.5, frameon=False)
    data = read(root / "profile" / "worker.json")
    for i, (key, offsets) in enumerate([("0", (-.08, .08)), ("5", (-.08, .08))]):
        ratios = []
        for j, pair in enumerate(data[key]["pairs"]):
            ratio = pair["ratio"] if "ratio" in pair else pair["graph"]["seconds"] / pair["stock"]["seconds"]
            ratios.append(ratio)
            worker.plot(ratio, i + offsets[j], "o", mfc="none", color=TEAL, ms=6)
        worker.scatter(np.median(ratios), i, marker="D", color=INK, s=36, zorder=3)
    worker.set_yticks([0, 1], ["Scherer A", "Scherer G"])
    worker.set_ylim(1.53, -.53)
    worker.set_xlim(.975, 1.02)
    worker.axvline(1, color=MUTED, ls="--", lw=.9)
    worker.set_xlabel("Complete session time, graph ÷ stock")
    for a in (kernel, worker):
        a.grid(axis="x", alpha=.2)
    fig.text(.02, .015, "Open circles show measured repeats. Diamonds show full-session medians. Lower time is better.",
             fontsize=8.5, color=MUTED)
    save(fig, out, "a100-profile")



def input_plot(data, out):
    """Use the first valid example per class under one person's identical setup."""
    sensors = ("C3", "Cz", "C4")
    tasks = ("Motor imagery", "Mental subtraction", "Word generation")
    with np.load(data, allow_pickle=False) as z:
        people = np.array([p.split("/")[-1] for p in z["subject"]])
        mask = (people == "P04") & (z["session"] == "1") & (z["interface"] == "Graz")
        rows = [np.flatnonzero(mask & (z["y"] == i))[0] for i in range(3)]
        electrode = [list(z["ch_names"]).index(s) for s in sensors]
        x = z["X"][rows][:, electrode]
    if x.shape != (3, 3, 480):
        raise ValueError("invalid official window example")
    fig, axes = plt.subplots(3, 3, figsize=(11.8, 6.1), sharex=True, sharey="row", layout="constrained")
    fig.get_layout_engine().set(rect=(.005, .05, .99, .94))
    fig.suptitle("What the raw-waveform decoder receives", x=.02, ha="left", fontsize=20, weight="bold")
    time = np.arange(480) / 120
    for col, title in enumerate(tasks):
        axes[0, col].set_title(title, loc="left", weight="bold")
        for row, sensor in enumerate(sensors):
            ax = axes[row, col]
            ax.plot(time, x[col, row], color=(BLUE, TEAL, GOLD)[row], lw=1)
            ax.axhline(0, color=MUTED, lw=.5, alpha=.45)
            ax.grid(axis="x", alpha=.15)
            ax.set_xlim(0, 4)
            if col == 0:
                ax.set_ylabel(f"{sensor} (µV)")
        axes[-1, col].set_xlabel("Seconds within input window")
    fig.text(.02, .015, "P04 · session 1 · Graz · first recorded example per class · 3 of 41 EEG channels · 120 Hz",
             fontsize=9, color=MUTED)
    save(fig, out, "official-eeg-input")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("eeg", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": "#ffffff",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "font.family": "DejaVu Sans",
                         "axes.edgecolor": "#c3cddb"})
    wave_plot(args.reports, args.output)
    runtime_plot(args.reports, args.output)
    input_plot(args.eeg, args.output)


if __name__ == "__main__":
    main()
