"""Reuse one verified Scherer em_half forward capture across the four causal tails."""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch

from diet_decomposition_patch import ARMS, patch
from diet_decomposition_tail import intervene
from quarter_niv import FIRST, PARENT_SHA
from returning_em_niv import CANDIDATE, PIN, validate_pair
from build_cb_speed import digest
from tape import tap


def check(model, n):
    if (model.log6["err"] or model.routeB.log["err"] or model.cf.log["err"] or model.cp.log["err"]
            or model.cp.nfail or model.cf.t != n or model.cp.t != n or model.team_a.age != n
            or model.team_a_graph.replays != n or model.team_a_graph.graph is None):
        raise ValueError("degraded NI-V em_half forward capture")


def live_a100(worker, package, channels, signals, rows, expected, meta, *, sources=None):
    source = package / "submission.py"
    original = source.read_text()
    if sources is None:
        sources = {arm: patch(original, settings) for arm, settings in ARMS.items()}
    for arm, text in sources.items():
        source.write_text(text)
        model = worker.fresh_model(worker.load(package, channels))
        labels = [int(model.predict(signals[r:r + 1].to("cuda:0"))[0]) for r in rows]
        if worker.recognised(model) != meta or model.dead:
            raise ValueError(f"{arm}: live A100 latch or inference failure")
        np.testing.assert_array_equal(labels, expected[arm], err_msg=f"{arm}: live A100 tape disagreement")
        print(f"PASS {arm}: live A100 Solver matches replay on all {len(rows)} labels", flush=True)
        del model


def score(args):
    sys.path.insert(0, str(Path.cwd() / "niv"))
    import worker

    smoke = args.mode == "tape-smoke"
    a100 = args.paired_a100
    expected_device = (8, 0) if a100 else (7, 5)
    if torch.cuda.get_device_capability() != expected_device:
        raise ValueError(f"expected an NI-V GPU with capability {expected_device}")
    overlay = Path("/enrolled.zip")
    if digest(CANDIDATE) != PARENT_SHA or digest(overlay) != args.enrolled_sha:
        raise ValueError("wrong NI-V archive or overlay")
    with np.load("/external.npz", allow_pickle=False) as data:
        signals = torch.from_numpy(data["X"])
        names, visits, channels = data["subject"], data["session"], data["ch_names"].tolist()
    runs = worker.orders.runs_from_manifest("/rows.tsv.gz", names, visits)
    stream = f"{sorted(set(names))[args.task]}|2"
    fresh = args.first is not None
    first = PIN if smoke else args.first if fresh else FIRST
    selected = ("c",) if fresh else tuple(ARMS)
    if args.sprint_pin is not None:
        import arm_c_sprint_policy as sprint
        selected = ("c", *sprint.ARMS)
    plans = [(s, i, first + i) for s in (("full",) if smoke else ("full", "within"))
             for i in (range(1) if smoke else range(8))]
    base, arms = {}, {arm: {} for arm in selected}
    with TemporaryDirectory() as tmp:
        package = Path(tmp)
        with ZipFile(CANDIDATE) as parent, ZipFile(overlay) as enrolled:
            parent.extractall(package)
            (package / "weights.npz").write_bytes(enrolled.read("weights.npz"))
        template = worker.load(package, channels)
        for scheme, index, seed in plans:
            rows = worker.orders.plans(names, visits, runs, scheme, seed)[stream]
            tag = f"{scheme}|{index}|{stream}"
            meta = {}

            def checked(model, n):
                check(model, n)
                meta["values"] = worker.recognised(model)

            start = perf_counter()
            items = tap(template, signals, rows, worker.fresh_model, checked, "cuda:0")
            rec, routed, err = meta["values"]
            flags = np.arange(len(rows)) >= rec
            if not 0 <= rec < len(rows):
                raise ValueError(f"{tag} did not recognise")
            np.testing.assert_array_equal(flags, [i.recognised for i in items])
            half, terminal, route = intervene(template, items, "half")
            for actual, field in ((half, "label"), (terminal, "terminal"), (route, "route_out")):
                np.testing.assert_array_equal(actual, np.asarray([getattr(i, field) for i in items]), err_msg=tag)
            outputs = {"half": half}
            for arm in selected:
                if args.sprint_pin is not None and arm != "c":
                    labels, _, _ = sprint.checked_replay(template, items, reference, arm)
                else:
                    reference = intervene(template, items, arm)
                    labels = reference[0]
                np.testing.assert_array_equal(labels[:rec], half[:rec], err_msg=f"{arm} {tag}")
                outputs[arm] = labels
            seconds = round(perf_counter() - start, 2)
            for arm, output in (("half", base), *arms.items()):
                output[tag + "|rows"] = np.asarray(rows, dtype=np.int64)
                output[tag + "|labels"] = outputs[arm]
                output[tag + "|meta"] = np.array([rec, routed, err, seconds], np.float64)
            print(f"PASS {tag}: {len(rows)} exact em_half route and terminal outputs, {seconds:.1f} s", flush=True)
            if smoke and a100:
                sources = None
                if args.sprint_pin is not None:
                    c_source = patch((package / "submission.py").read_text(), ARMS["c"])
                    sources = {"c": c_source, **{arm: sprint.patch(c_source, arm) for arm in sprint.ARMS}}
                live_a100(worker, package, channels, signals, rows, outputs, meta["values"], sources=sources)

    person = args.task
    filename = f"smoke_{person}.npz" if smoke else f"p{person}_c{person % 4}.npz"
    lead = (args.pinned / "active" / f"ret_trial_k{person}" if smoke else
            args.baseline / f"ret_lead_k{person}") / filename
    with TemporaryDirectory(dir=args.root) as tmp:
        folder = Path(tmp)
        control = folder / "half.npz"
        np.savez(control, **base)
        if fresh:
            # Fresh orders have no archived labels. The per-stream Solver identity above is the control.
            pass
        elif a100:
            with np.load(lead, allow_pickle=False) as pinned:
                if pinned.files != list(base):
                    raise ValueError("A100 stream keys differ from the Turing reference")
                for key in pinned.files:
                    if key.endswith("|rows"):
                        np.testing.assert_array_equal(base[key], pinned[key], err_msg=key)
                    if key.endswith("|meta"):
                        np.testing.assert_array_equal(base[key][:3], pinned[key][:3], err_msg=key)
        elif any(v["changed"] for v in validate_pair(lead, control).values()):
            raise ValueError("NI-V forward capture changed pinned em_half labels")
        for arm, output in arms.items():
            path = folder / f"{arm}.npz"
            np.savez(path, **output)
            if args.sprint_pin is not None and not smoke and arm == "c":
                pinned_c = args.sprint_pin / f"ret_trial_k{person}" / filename
                if any(v["changed"] for v in validate_pair(pinned_c, path).values()):
                    raise ValueError("Scherer arm c changed pinned A100 labels")
            if smoke and not a100:
                direct = args.root.parent / "niv-smoke" / arm / f"ret_trial_k{person}" / filename
                if any(v["changed"] for v in validate_pair(direct, path).values()):
                    raise ValueError(f"{arm}: causal tail differs from the live Solver")
            print(f"PASS {arm} {args.mode} person {person}: {validate_pair(control, path)}", flush=True)
        for arm in (("half", *selected) if a100 or fresh else selected):
            out = args.root / arm / f"ret_trial_k{person}" / filename
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                raise FileExistsError(out)
            (folder / f"{arm}.npz").rename(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tape-smoke", "tape-score"))
    parser.add_argument("task", type=int)
    parser.add_argument("enrolled_sha")
    parser.add_argument("--paired-a100", action="store_true")
    parser.add_argument("--sprint-pin", type=Path, help="enable the arm c sprint and pin A100 control shards")
    parser.add_argument("--first", type=int, help="first fresh order for a paired em_half and c release")
    for name in ("baseline", "pinned", "root"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.first is not None and (args.mode != "tape-score" or args.first < FIRST + 8):
        parser.error("--first requires tape-score and an order after the pinned release")
    if args.sprint_pin is not None and (not args.paired_a100 or args.first is not None):
        parser.error("--sprint-pin requires the paired A100 71864 release")
    score(args)


if __name__ == "__main__":
    main()
