"""Compare FBLight variants across their signed evaluation instruments."""
import argparse
from pathlib import Path

from adaptation import interval, layout
from native_adaptation import locked
from plot import BLUE, MUTED, save


def plot(receipts, output):
    dev = locked(receipts / "gate-dev.json",
                 "59bef3c831923f0a2ddf7d45f8878647a6703649d83c3b407da4b821fee7cbcd")["scores"]
    external = locked(receipts / "always-external.json",
                      "a60b05b91657d9fe11ea3b67d4dcd4609f1cf5c981ba78e6c280b0e0f0de6750")
    external_gate = locked(receipts / "gate-external.json",
                           "13ffeb61bfaad56d23598d68cbd7f048ee61149027ac7a9570a924cb2f07e748")
    contrasts = ("fbl2_minus_stock", "selected_minus_stock")
    rows = [
        (label, tuple(dev[order]["comparisons"][key] for key in contrasts))
        for order, label in (("within0", "Held-new · within0"), ("full0", "Held-new · full0"))
    ]
    returning = dev["returning"]["comparisons"]["weighted_20_80"]
    rows += [
        ("Returning · P04/P10 at 20/80", tuple(returning[k] for k in contrasts)),
        ("External · Scherer, two orders", (external["contrast"], external_gate["versus_stock"])),
    ]
    fig, (ax,) = layout("FBLight showed a gain on the held-new replays")
    for row, (_, (always, protected)) in enumerate(rows):
        interval(ax, always, row - .13, MUTED, "Always doubled" if row == 0 else None)
        interval(ax, protected, row + .13, BLUE, "Stock-protected rule" if row == 0 else None)
    ax.axhline(1.5, color="#d7e1e9", lw=1)
    ax.axhline(2.5, color="#d7e1e9", lw=1)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.set(xlim=(-.55, 1.0), ylim=(3.58, -.58))
    ax.set_title("Each variant vs the exact stock path · paired 80% intervals",
                 loc="left", fontsize=10, color=MUTED)
    ax.legend(loc="lower right", frameon=False, ncol=1)
    fig.text(.028, .024, "Full-system replays · two known returning people · Scherer repeats previously evaluated participants",
             fontsize=8.8, color=MUTED)
    save(fig, output, "fblight-across-cohorts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot(args.receipts, args.output)
