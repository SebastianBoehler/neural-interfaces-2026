"""Compare the motor-supervised waveform component in the complete stack."""
import argparse
from pathlib import Path

from adaptation import interval, layout
from native_adaptation import locked
from plot import BLUE, GOLD, MUTED, save


def plot(receipts, output):
    held = locked(receipts / "held.json",
                  "58034507675e38823a563967ea2a19cb31d9a9ff64b30f1558923b42e7d3ac6a")["reports"]
    external = locked(receipts / "external.json",
                      "e5e2ccea9e101c3359325a5c15c2f14ef223c5b59290381f8d6320b56089e549")["contrasts"]
    rows = (
        ("Held-new · within0", held["within0"]["contrast"]),
        ("Held-new · full0", held["full0"]["contrast"]),
        ("External · Scherer", external),
    )
    fig, (ax,) = layout("Motor supervision gave mixed results in the full system")
    for y, (_, data) in enumerate(rows):
        interval(ax, data["joint_minus_stock"], y-.12, BLUE, "Versus cap250" if y == 0 else None)
        interval(ax, data["joint_minus_hard"], y+.12, GOLD, "Versus matched waveform model" if y == 0 else None)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.set(xlim=(-.70, .68), ylim=(2.65, -.58))
    ax.axhline(1.5, color="#d7e1e9", lw=1)
    ax.legend(loc="lower left", bbox_to_anchor=(.01, 1.01), ncol=2, frameon=False)
    fig.text(.028, .024, "Paired 80% intervals · 10 held-new PROTEUS people · 9 Scherer people in two fresh delivery orders",
             fontsize=8.8, color=MUTED)
    save(fig, output, "motor-supervision-across-cohorts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.receipts, args.output)
