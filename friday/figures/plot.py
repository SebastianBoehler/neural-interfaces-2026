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
    ("motor", "PhysioNet transfer", "motor_transfer"),
    ("motor", "Shuffled motor source", "motor_null"),
    ("raw", "Raw supervised", "raw_supervised"),
    ("raw", "Masked latent", "raw_masked"),
    ("style", "Synthetic covariance", "raw_style_full"),
    ("style", "Matched sensor gains", "raw_style_diagonal"),
    ("norm", "Fixed normalization", "raw_norm_fixed"),
    ("norm", "Learned normalization", "raw_norm_learned"),
)
PEOPLE = [f"P{i:02}" for i in range(1, 11)]


def read(path):
    return json.loads(path.read_text())


def save(fig, out, stem):
    fig.savefig(out / (stem + ".png"), dpi=170, facecolor=BG)
    fig.savefig(out / (stem + ".pdf"), facecolor=BG)
    plt.close(fig)


def waveform_row(data, label, seed, full_contrast, whole_arm):
    if abs(data["absolute"]["stock"]["pp"] - STOCK) > 1e-8:
        raise ValueError("mismatched complete-stock score")
    return label, seed, data["contrasts"][full_contrast], data["absolute"][whole_arm]["pp"]


def band_plot(root, out):
    paths = [root / "band" / f"band_{seed}_result.json" for seed in SEEDS]
    if not all(path.is_file() for path in paths):
        print("filter-bank initialization: waiting for a complete seed pair")
        return
    results = [read(path) for path in paths]
    for seed, result in zip(SEEDS, results, strict=True):
        if (result["seed"], result["source_gpu"], result["historical_f7_f9_gpu"]) != (seed, "A100", "H100"):
            raise ValueError("mismatched filter-bank seed or training GPU")
        if abs(result["absolute"]["stock"]["pp"] - STOCK) > 1e-8:
            raise ValueError("mismatched complete-stock score")
    anchor = results[0]["historical_s1c"]
    if any(result["historical_s1c"]["sha256"] != anchor["sha256"] for result in results):
        raise ValueError("mismatched historical filter-bank reference")

    fig, (whole, people) = plt.subplots(1, 2, figsize=(11.6, 6.2), layout="constrained")
    fig.get_layout_engine().set(rect=(.015, .06, .98, .94))
    fig.suptitle("Trainable filter-bank initialization on held EEG", x=.025, ha="left",
                 fontsize=20, weight="bold")
    whole.set_title("Whole-decoder balanced accuracy", loc="left", weight="bold")
    people.set_title("Spectral minus random initialization", loc="left", weight="bold")
    arms = ("spectral_genuine", "random_genuine", "f7", "f9")
    for seed, result, color, marker, offset in zip(SEEDS, results, (BLUE, GOLD), ("o", "D"), (-.11, .11), strict=True):
        values = [result["absolute"][arm]["pp"] for arm in arms]
        whole.scatter(values, np.arange(len(arms)) + offset, color=color, marker=marker,
                      label=f"Seed {seed}", s=44)
        contrast = result["contrasts"]["spectral_genuine-random_genuine"]
        mean = contrast["pp"]
        low, high = contrast["paired80"]
        people.errorbar(mean, offset, xerr=[[mean - low], [high - mean]], color=color,
                        fmt=marker, ms=5, lw=1.5, capsize=3)
        people.scatter([contrast["people"][name] for name in PEOPLE],
                       np.arange(1, len(PEOPLE) + 1) + offset, color=color, marker=marker, s=30)
    whole.scatter(anchor["score"], len(arms), color=MUTED, marker="|", s=130)
    whole.set_yticks(range(len(arms) + 1), ["F12 spectral · A100", "F12 random · A100",
                                          "F7 supervised · H100", "F9 fixed norm · H100",
                                          "Historical filter bank"])
    whole.set_ylim(len(arms) + 1.0, -.48)
    whole.set_xlabel("Balanced accuracy (%)")
    whole.legend(loc="lower left", frameon=False, fontsize=8.5, ncol=2)
    people.set_yticks(range(len(PEOPLE) + 1), ["Pooled (paired 80%)", *PEOPLE])
    people.set_ylim(len(PEOPLE) + .48, -.55)
    people.set_xlabel("Whole-decoder change (percentage points)")
    people.axvline(0, color=MUTED, ls="--", lw=.85)
    for axis in (whole, people):
        axis.grid(axis="x", alpha=.19)
        axis.set_axisbelow(True)
    fig.text(.025, .02, "18 Nov 2026  ·  10 fit-excluded PROTEUS people  ·  8 pinned orders  ·  Historical results are reference points",
             fontsize=8.6, color=MUTED)
    save(fig, out, "held-filter-initialization")


def wave_plot(root, out):
    rows = []
    for folder, label, name in MODELS:
        paths = [root / folder / f"{name}_{seed}_result.json" for seed in SEEDS]
        if not all(path.is_file() for path in paths):
            if folder in ("prototype", "time"):
                raise FileNotFoundError(f"incomplete reference results for {name}")
            print(f"{name}: waiting for a complete seed pair")
            continue
        for s, path in zip(SEEDS, paths, strict=True):
            rows.append(waveform_row(read(path), label, s, "full_genuine-stock", "wave_genuine"))
    scale_paths = [root / "scale" / f"input_scale_{seed}_result.json" for seed in SEEDS]
    if all(path.is_file() for path in scale_paths):
        scale_results = [read(path) for path in scale_paths]
        if tuple(result["seed"] for result in scale_results) != SEEDS:
            raise ValueError("mismatched sensor scale seeds")
        for fit, label in (("genuine", "Sensor scale · own history"), ("donor", "Sensor scale · donor training")):
            arm = f"{fit}_trained_genuine"
            for seed, data in zip(SEEDS, scale_results, strict=True):
                rows.append(waveform_row(data, label, seed, f"full_{arm}-stock", f"wave_{arm}"))
    else:
        print("sensor scaling: waiting for a complete seed pair")
    heat = np.array([[r[2]["people"][p] for p in PEOPLE] for r in rows])
    n = len(rows)
    heat_limit = max(2.5, float(np.ceil(np.abs(heat).max())))
    fig = plt.figure(figsize=(12.5, max(7.4, n * .39 + 2.0)), layout="constrained")
    fig.get_layout_engine().set(rect=(.005, .04, .99, .955))
    ax = fig.subplots(1, 3, width_ratios=[1.0, 2.05, .66], sharey=True)
    fig.suptitle("Waveform decoders on held PROTEUS people", x=.022, ha="left",
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
    img = ax[1].imshow(heat, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0, vmin=-heat_limit, vmax=heat_limit),
                       aspect="auto", interpolation="nearest")
    cb = fig.colorbar(img, ax=ax[1], location="bottom", shrink=.8, pad=.06)
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
    kernel.set_title("Isolated kernel latency", loc="left", fontsize=12.5, weight="bold")
    worker.set_title("Full cap250 session time", loc="left", fontsize=12.5, weight="bold")
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
    band_plot(args.reports, args.output)
    runtime_plot(args.reports, args.output)
    input_plot(args.eeg, args.output)


if __name__ == "__main__":
    main()
