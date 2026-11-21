"""Render preregistered gates and tables from the Leonardo diagnostic receipt."""
import argparse
import json
from pathlib import Path

import numpy as np


def hedge(summary):
    scores = np.asarray(summary["cell_weighted"])
    deciles = scores[:, 2] - scores[:, 0]
    sessions = np.asarray(list(summary["session"].values()))
    gains = sessions[:, 2] - sessions[:, 0]
    return dict(decile_span=float(np.ptp(deciles)), first_minus_last=float(deciles[0] - deciles[-1]),
                session_span=float(np.ptp(gains)), mixed_sign=bool(gains.min() <= 0 < gains.max()))


def gates(result):
    missed = [a for a in result["arbiter"] if a["oracle_swap"] and a["first_trigger"] is None]
    false = [a for a in result["arbiter"] if not a["oracle_swap"] and a["first_trigger"] is not None]
    correct = [a for a in result["arbiter"] if a["oracle_swap"] and a["first_trigger"] is not None]
    summary = {k: hedge(v) for k, v in result["deciles"].items()}
    position = len(summary) == 2 and all(x["decile_span"] >= 1 for x in summary.values())
    sessions = len(summary) == 2 and all(x["session_span"] >= 1 and x["mixed_sign"] for x in summary.values())
    auc = result["residual"]["new"]["confused"]["auc"]
    return dict(P1=auc is not None and auc >= 0.60,
                P2=bool(correct and not missed and not false),
                P3=(position or sessions) if summary else None,
                arbiter=dict(true=len(correct), missed=len(missed), false=len(false)), hedge=summary)


def show(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def auc_line(label, r):
    fields = [f'{show(r[k]["auc"], 3)} ({r[k]["n"]})' for k in ("all", "correct", "confused")]
    return "| " + " | ".join((label, *fields)) + " |"


def arbiter_line(case):
    names = ("M", "S", "W")
    permutation = "".join(names[i] for i in case["oracle"])
    if case["tied_nonidentities"]:
        permutation += "*"
    fields = [case["fold"], case["stream"], str(case["order"]), permutation, f'{case["oracle_gain"]:+.2f}']
    fields += [show(x) for x in case["T"].values()]
    fields += [show(case["enrolled_gap"]), str(case["first_trigger"] or "—")]
    return "| " + " | ".join(fields) + " |"


def render(result):
    lines = [
        "# Fable Round 2 · diagnostic tables",
        "",
        "Full session shuffles 71124–71131 on production-20261120-32a94ab7. Held-new covers ten people.",
        "Returning covers P04/P10 visits 2–3. Each order repeats the same people and can repeat individual errors.",
        "",
        "## P1 · Word-higher AUC",
        "",
        "The score is the six-channel raw-window contrast, median/IQR-standardised and clipped per session.",
        "The denominator in parentheses counts deliveries. 'Confused' requires a true and predicted non-MI label.",
        "",
        "| Cohort / stream | All S/W | Lead correct | Lead confused |",
        "|---|---:|---:|---:|",
    ]
    for cohort, label in (("new", "Held-new"), ("ret", "Returning")):
        lines.append(auc_line(f"**{label}**", result["residual"][cohort]))
        lines.extend(auc_line(f"{label} {stream}", a) for stream, a in result["residual_sessions"][cohort].items())

    lines += [
        "", "## P2 · Session oracle and causal arbiter", "",
        "M = motor imagery, S = subtraction, W = word. The oracle maximises plain session accuracy.",
        "An asterisk marks a tie between nonidentity maps. Identity wins ties with identity.",
        "T uses the enrolled gap sign when its magnitude is at least 0.30. Smaller gaps display raw T and block firing.",
        "",
        "| Fold | Stream | Order | Label map | Oracle gain, pp | T60 | T120 | T end | Visit-1 gap | Trigger age |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(arbiter_line(case) for case in result["arbiter"])

    for cohort, label in (("new", "Held-new"), ("ret", "Returning")):
        if cohort not in result["deciles"]:
            continue
        r = result["deciles"][cohort]
        lines += [
            "", f"## P3 · {label} delivery deciles", "",
            "A = hard-only terminal. B = Team B encoder with causal alignment and feature normalisation.",
            "Lead = hard + Team B nine25 terminal. Weighted accuracy uses full-session cell/class counts.",
            "",
            "| Decile | A | B | Lead | A weighted | B weighted | Lead weighted | Weighted gain, pp |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for j, (a, w) in enumerate(zip(r["accuracy"], r["cell_weighted"], strict=True), 1):
            lines.append(f'| {j} | ' + ' | '.join(show(x) for x in a + w) + f' | {w[2]-w[0]:+.2f} |')
        lines += ["", "| Session | Hard only | Team B | Lead | Gain, pp |", "|---|---:|---:|---:|---:|"]
        lines.extend(f"| {s} | " + " | ".join(show(v) for v in a) + f" | {a[2]-a[0]:+.2f} |"
                     for s, a in r["session"].items())
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    result = json.loads(args.diagnostic.read_text())
    print(json.dumps(gates(result), indent=2, allow_nan=False))
    if args.table:
        args.table.write_text(render(result))


if __name__ == "__main__":
    main()
