"""Plot independently signed held-new and external factorial comparisons."""
import argparse
from pathlib import Path

from adaptation import interval, layout
from native_adaptation import locked
from plot import BLUE, GOLD, MUTED, TEAL, save


def plot(held, external, output):
    rows = (
        ("Held-new · within0", held["orders"]["within0"]["contrasts"], "joint"),
        ("Held-new · full0", held["orders"]["full0"]["contrasts"], "joint"),
        ("External · Scherer", external["contrasts"], "combo"),
    )
    fig, (gains, ceiling) = layout("Combining FBLight and waveform lost ground on Scherer", width=13, columns=2)
    labels = ("Combo vs cap250", "Combo vs FBLight alone", "Combo vs waveform alone")
    colors = BLUE, GOLD, TEAL
    for y, (_, contrasts, prefix) in enumerate(rows):
        offsets = (-.19, 0, .19)
        parents = ("stock", "FBL2" if prefix == "joint" else "fbl2", "hard")
        for parent, dy, color, label in zip(parents, offsets, colors, labels, strict=True):
            interval(gains, contrasts[f"{prefix}_minus_{parent}"], y+dy, color, label if y == 0 else None)
        interval(ceiling, contrasts["interaction"], y-.12, MUTED, "Measured interaction" if y == 0 else None)
        interval(ceiling, contrasts["parent_oracle"], y+.12, BLUE, "Label-aware rescue ceiling" if y == 0 else None)
    gains.set_yticks(range(len(rows)), [name for name, _, _ in rows])
    for ax in (gains, ceiling):
        ax.set_ylim(2.65, -.65)
        ax.axhline(1.5, color="#d7e1e9", lw=1)
        ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1, 1))
    fig.text(.028, .024, "Paired 80% intervals · 10 held-new PROTEUS people · 9 reused Scherer people in two fresh orders",
             fontsize=8.8, color=MUTED)
    save(fig, output, "fblight-waveform-factorial")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--held", type=Path, required=True)
    parser.add_argument("--held-sha256", required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--external-sha256", required=True)
    args = parser.parse_args()
    plot(locked(args.held, args.held_sha256), locked(args.external, args.external_sha256), args.output)
