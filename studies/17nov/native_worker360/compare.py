"""Report model-label and work parity before interpreting timing."""
import json
from pathlib import Path
import sys


def main(suite, *modes):
    failures = []
    details = {}
    for mode in modes:
        reference = json.loads(Path(f"reference_{mode}.json").read_text())
        native = json.loads(Path(f"native_{mode}.json").read_text())
        if (reference["torch"], reference["gpu"]) != (native["torch"], native["gpu"]):
            raise ValueError("reference and native devices differ")
        cases = {}
        if reference["streams"].keys() != native["streams"].keys():
            raise ValueError("reference and native rounds differ")
        if suite == "smoke":
            for arm, results in (("reference", reference), ("native", native)):
                for name, first in results["streams"]["first"].items():
                    repeat = results["streams"]["repeat"][name]
                    if any(first[field] != repeat[field] for field in ("labels", "activity")):
                        failures.append(f"{mode}/{arm}/{name}/fresh-repeat")
        for repeat, rows in reference["streams"].items():
            counterpart = native["streams"][repeat]
            if rows.keys() != counterpart.keys():
                raise ValueError("reference and native stream sets differ")
            for name, a in rows.items():
                b = counterpart[name]
                if len(a["labels"]) != len(b["labels"]):
                    raise ValueError("label count differs")
                changed = [i for i, (x, y) in enumerate(zip(a["labels"], b["labels"])) if x != y]
                case = {"windows": len(a["labels"]), "reference_s": a["s"], "native_s": b["s"],
                        "changed_labels": len(changed), "first_changed": changed[:10],
                        "reference_activity": a["activity"], "native_activity": b["activity"]}
                cases[f"{repeat}/{name}"] = case
                if changed or a["activity"] != b["activity"]:
                    failures.append(f"{mode}/{repeat}/{name}")
        ref_s = sum(c["reference_s"] for c in cases.values())
        nat_s = sum(c["native_s"] for c in cases.values())
        details[mode] = {"cases": cases, "reference_replay_s": ref_s, "native_replay_s": nat_s,
                         "ratio": nat_s/ref_s, "reference_load_s": reference["load_s"],
                         "native_load_s": native["load_s"],
                         "reference_wall_s": reference["wall_s"], "native_wall_s": native["wall_s"],
                         "reference_rss_kib": reference["rss_kib"], "native_rss_kib": native["rss_kib"],
                         "reference_cuda_live": reference["cuda_live_bytes"],
                         "native_cuda_live": native["cuda_live_bytes"],
                         "reference_cuda_reserved": reference["cuda_reserved_bytes"],
                         "native_cuda_reserved": native["cuda_reserved_bytes"]}
    status = {"suite": suite, "parity": not failures, "failures": failures, **details}
    Path("comparison.json").write_text(json.dumps(status, indent=2)+"\n")
    if failures:
        raise ValueError(f"worker parity failed in {failures}")
    print("NATIVE_A100_MATCHED_WORK_PARITY_PASS", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
