"""Summarize native trace attribution and evaluate the frozen terminal offsets."""
import hashlib
import json

from errors import table
from oracles import ceiling_report, freeze
from stage_report import calibration, recognition, stage_report
from tapes import ROSTERS, stages
from terminal import fit, held

FIT = (71800, 71801, 71802, 71803)
HELD = (71808, 71809, 71810, 71811)


def component(work, data, interfaces, part):
    selected = work / "terminal_selection.json"
    if part == "fit":
        result = {roster: fit(stages(data, work, roster, FIT), roster) for roster in ROSTERS}
        freeze(selected, {"orders": FIT, "rosters": result})
        return json.dumps(result, indent=2)
    if part == "held":
        with selected.open() as f:
            result = json.load(f)
        if tuple(result["orders"]) != FIT or set(result["rosters"]) != set(ROSTERS):
            raise ValueError("wrong terminal freeze receipt")
        rows = [row for r in ROSTERS
                for row in held(stages(data, work, r, HELD), r, result["rosters"][r], HELD)]
        digest = hashlib.sha256(selected.read_bytes()).hexdigest()
        return (table(["Stack", "Terminal test", "Frozen setting", "Fit J Δ", "New Δ [80%]", "Ret. Δ [80%]",
                       "50/50 Δ [80%]", "Changed new/ret"], rows) + f"\n\nFreeze receipt SHA-256 `{digest}`.")
    blocks = []
    for roster in ROSTERS:
        orders = FIT + HELD
        cases = stages(data, work, roster, orders)
        blocks.extend([f"### {roster}", stage_report(cases, interfaces, roster == "ba_six"),
                       "**Calibration and class bias**", calibration(cases, interfaces),
                       "**Recognition**", recognition(cases),
                       "**Retrospective diagnostics**", ceiling_report(cases, orders, interfaces)])
    return "\n\n".join(blocks)
