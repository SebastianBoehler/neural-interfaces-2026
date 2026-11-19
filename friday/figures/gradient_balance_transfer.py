"""Plot gradient balance on standalone raw models and complete cap250."""

import argparse
from pathlib import Path

from adaptation import interval, layout
from native_adaptation import locked
from plot import BLUE, GOLD, INK, MUTED, TEAL, save


STANDALONE = "d22fcd1b314ac37cc3e6ee42b53b580e124209c08d77550db5df6ea91790a5ee"
HELD = "8fb06208095e6ea43598440884a67bb3ae9a71241a827111235dd2c9d5e20430"
EXTERNAL = "2c36e271238f69e750edd2f4d470d69efe8d10e84133dca48639d912be213327"
STAGES = "cd3780d6928782f1049e415394771bb3a81fcfd8b80101104d8438edbfb538d6"
CONTRASTS = (
    ("balance_minus_hard", BLUE, -.20, "Balance vs hard"),
    ("mirrored_minus_hard", GOLD, 0, "Mirror vs hard"),
    ("balance_minus_mirrored", TEAL, .20, "Balance vs mirror"),
)


def bounded_comparisons(seed):
    return {
        key: {"pp": value["mean"], "paired80": value["interval"]}
        for name, value in seed["paired"].items()
        if (key := name.removeprefix("bounded_")) != name
    }


def plot(receipts):
    standalone = locked(receipts / "standalone.json", STANDALONE)["seeds"]
    held = locked(receipts / "held.json", HELD)["orders"]
    external = locked(receipts / "external.json", EXTERNAL)["contrasts"]
    groups = (
        ("Raw only · seed 0", bounded_comparisons(standalone["0"])),
        ("Raw only · seed 1", bounded_comparisons(standalone["1"])),
        ("Cap250 · within0", held["within0"]["contrasts"]),
        ("Cap250 · full0", held["full0"]["contrasts"]),
        ("Cap250 · Scherer", external),
    )
    fig, (ax,) = layout("Raw decoder gains weakened inside cap250", width=11)
    ax.axhspan(-.52, 1.5, color="#edf3f8", zorder=0)
    for row, (_, data) in enumerate(groups):
        for key, color, offset, label in CONTRASTS:
            effect = data[key]
            interval(ax, effect, row + offset, color, label if row == 0 else None)
            ax.annotate(f"{effect['pp']:+.2f}", (effect["paired80"][1], row + offset),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK)
    ax.set_yticks(range(len(groups)), [name for name, _ in groups])
    low = min(data[key]["paired80"][0] for _, data in groups for key, *_ in CONTRASTS)
    high = max(data[key]["paired80"][1] for _, data in groups for key, *_ in CONTRASTS)
    ax.set(xlim=(min(-.55, low - .10), max(1.60, high + .19)), ylim=(len(groups) - .49, -.52))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False, fontsize=9)
    fig.text(.028, .024, "Paired 80% intervals · matched A100 source fits · 10 held-new people · 9 reused external people",
             fontsize=8.5, color=MUTED)
    save(fig, receipts, "gradient-balance-transfer")


def plot_stages(receipts):
    results = locked(receipts / "stages.json", STAGES)["orders"]
    fig, panels = layout("Later cap250 stages deepened the within0 loss", width=11.4, columns=2)
    stages = {"raw": "After native routing", "filtered": "After CountFilter", "terminal": "After CSP fusion"}
    for ax, order in zip(panels, ("within0", "full0"), strict=True):
        ax.set_title(order, loc="left", fontsize=11, weight="bold", color=INK)
        ax.axhspan(1.54, 2.5, color="#edf3f8", zorder=0)
        for row, name in enumerate(stages):
            data = results[order]["stages"][name]["contrasts"]
            for key, color, offset, label in CONTRASTS[:2]:
                interval(ax, data[key], row + offset + .10, color, label if row == 0 else None)
        ax.set(xlim=(-.58, .43), ylim=(2.49, -.48))
    panels[0].set_yticks(range(3), list(stages.values()))
    panels[1].legend(loc="lower right", bbox_to_anchor=(1, 1.02), ncol=2, frameon=False, fontsize=9)
    fig.text(.028, .024,
             "Paired 80% intervals · 10 held-new people · stage readouts from the original complete native replays",
             fontsize=8.5, color=MUTED)
    save(fig, receipts, "gradient-balance-stages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    args = parser.parse_args()
    plot(args.receipts)
    plot_stages(args.receipts)
