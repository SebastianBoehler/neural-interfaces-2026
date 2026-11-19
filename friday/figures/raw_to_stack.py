"""Compare matched-hard bounds for signed raw and complete cap250 training waves."""

import argparse
from pathlib import Path

from adaptation import interval, layout
from native_adaptation import locked
from plot import BLUE, GOLD, INK, MUTED, TEAL, save


RECEIPTS = {
    "motor_raw": "1b9dd9d99b655d1dd5f1fff3655995d26593e6373bab6729fb64107b46daf5df",
    "motor_full": "58034507675e38823a563967ea2a19cb31d9a9ff64b30f1558923b42e7d3ac6a",
    "balance_raw": "d22fcd1b314ac37cc3e6ee42b53b580e124209c08d77550db5df6ea91790a5ee",
    "balance_full": "8fb06208095e6ea43598440884a67bb3ae9a71241a827111235dd2c9d5e20430",
    "crossfade_raw": "86b65abf6273b296f5ef4fd8e4ce4712c8a1e7794aa16a3fb4c2bd31f6a7d08e",
    "crossfade_full": "4552919148d10ad3af6896249c2658bd2d75b786585993cc93937c40423d4643",
}
ROWS = (
    ("Motor supervision", "motor", "joint_minus_detached", "joint_minus_hard"),
    ("Gradient balance", "balance", "balance_minus_hard", "balance_minus_hard"),
    ("Same-trial crossfade", "crossfade", "same_trial_minus_hard", "same_trial_minus_hard"),
    ("Cross-trial crossfade", "crossfade", "cross_trial_minus_hard", "cross_trial_minus_hard"),
)


def plot(folder):
    records = {name: locked(folder / f"{name}.json", sha) for name, sha in RECEIPTS.items()}
    fig, (raw, full) = layout("Training gains weakened in the complete decoder", width=11.8, columns=2)
    raw.set_title("Standalone raw · bounded inference", loc="left", color=INK, fontsize=11, weight="bold")
    full.set_title("Complete cap250 · corrected replay", loc="left", color=INK, fontsize=11, weight="bold")
    for row, (_, source, raw_key, full_key) in enumerate(ROWS):
        standalone = records[f"{source}_raw"]["seeds"]
        complete = records[f"{source}_full"]
        for seed, order, raw_color, full_color, shift in zip(
            ("0", "1"), ("within0", "full0"), (BLUE, GOLD), (TEAL, INK), (-.13, .13), strict=True
        ):
            pair = standalone[seed]["paired"][f"bounded_{raw_key}"]
            interval(raw, dict(pp=pair["mean"], paired80=pair["interval"]), row + shift, raw_color,
                     f"Seed {seed}" if row == 0 else None)
            if source == "motor":
                result = complete["reports"][order]["contrast"][full_key]
            else:
                result = complete["orders"][order]["contrasts"][full_key]
            interval(full, result, row + shift, full_color, order if row == 0 else None)
    raw.set_yticks(range(len(ROWS)), [name for name, *_ in ROWS])
    for ax in (raw, full):
        ax.set(xlim=(-.81, 1.55), ylim=(3.5, -.5))
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.12), ncol=2, frameon=False, fontsize=8.5)
    fig.text(.028, .024,
             "Paired 80% intervals · pupil fits exclude each held person · cap250 combines both seeds · H100 motor fits, A100 other fits",
             fontsize=8.5, color=MUTED)
    save(fig, folder, "raw-to-stack")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    plot(p.parse_args().folder)
