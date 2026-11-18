"""Authenticate the four returning native stage banks without opening held gold."""

import json
from pathlib import Path

import numpy as np

from verify_smoke import check, sha


BASE = Path("/mnt/lustre/work/gehler/gfh527")
PRODUCER = BASE / "ni26_friday_returning_native_inner_v2_20261119"
PRODUCER_SHA = "3f513cf5e0fd9f2db3cfdb15c9f09ae1d09e8bbbf9092b0f2e3176444fed8626"
SMOKE = "7574a7386c38a324b7bb0d65cf86df704a71f4a6944698cc2c16f45a7416f07c"
REPO = BASE / "neural-interfaces-2026"
VISITS = (("P04", "2", 189), ("P04", "3", 188), ("P10", "2", 186), ("P10", "3", 199))
ARMS = ("native_zero", "recording", "causal")


def collect():
    if (sha(PRODUCER / "source.sha256") != PRODUCER_SHA
            or sha(PRODUCER / "smoke/native_inner.json") != SMOKE):
        raise ValueError("independent native producer or smoke source differs")
    check(PRODUCER)
    with np.load(REPO / "data/proteus_windows.npz", allow_pickle=False) as data:
        people, days = data["subject"], data["session"]
    with np.load(REPO / "runs/shuffle_perms.npz", allow_pickle=False) as z:
        order = z["full|0|seen_session"]
    found, records = [], {}
    for k, (who, day, n) in enumerate(VISITS):
        part = f"visit{k}"
        rec = check(PRODUCER, part)
        root = PRODUCER / part
        if (rec["visit"], rec["fold"], rec["n"]) != (f"{who}|{day}", f"{who}_s23", n):
            raise ValueError("native successor operated on the wrong fit or visit")
        rows = order[(people[order] == f"Dreyer2026Proteus/{who}") & (days[order] == day)]
        with np.load(root / "native_inner.npz", allow_pickle=False) as z:
            np.testing.assert_array_equal(z["rows"], rows)
            expected_ages = ([20] if (who, day) == ("P10", "2") else [])
            for arm in ARMS:
                slots = rec["arms"][arm]
                labels, scores = z[f"{arm}|labels"], z[f"{arm}|scores"]
                if (len(labels) != n or scores.shape != (n, 3) or slots["refit_ages"] != expected_ages
                        or slots["steps"] != 20 * len(expected_ages)):
                    raise ValueError("slot coverage, real actor or warm refit ages changed")
                np.testing.assert_array_equal(labels, scores.argmax(1))
                for age in slots["refit_ages"]:
                    np.testing.assert_array_equal(z[f"{arm}|target_rows|{age}"], rows[:age])
            count = int(np.count_nonzero(z["recording|labels"] != z["causal|labels"]))
            delta = float(np.abs(z["recording|scores"] - z["causal|scores"]).max())
            if not np.isfinite(delta):
                raise ValueError("incomplete real terminal posterior")
        key = f"{who}|{day}"
        records[key] = dict(rows=n, first_global_row=int(rows[0]), receipt=sha(root / "native_inner.json"),
                            bank=rec["output"], max_terminal_posterior_delta=delta, changed_final_decisions=count,
                            native_mix_delta=rec["native_mix_delta"], strict_target_delta=rec["strict_target_delta"],
                            warm_family_delta=rec["warm_family_delta"], arms=rec["arms"])
        found.extend(rows.tolist())
    np.testing.assert_array_equal(np.sort(found), np.sort(order))
    result = dict(order="full0", source=sha(Path("source.sha256")), producer=PRODUCER_SHA,
                  smoke=SMOKE, windows=len(found), held_labels_opened=False, accuracy_computed=False,
                  scope="conditional native source fit with production rest, eye, core2, event priors and old-OOF exposure",
                  records=records)
    target = Path("returning_native_inner.json")
    with target.open("x") as file:
        json.dump(result, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    target.chmod(0o444)
    print("PASS four returning visits and 762 native decision rows", sha(target), flush=True)


if __name__ == "__main__":
    collect()
