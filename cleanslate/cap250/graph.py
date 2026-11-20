"""Emit or submit the cap250 Slurm DAG. Planning and sbatch submission use only Python's stdlib."""

import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import sys

from components import Family, FAMILIES, TTT_FORMS, Variant, part_path
from route_inputs import b2_fits, subfold

ROOT = Path(__file__).resolve().parents[2]
HERE = "cleanslate/cap250"
HARNESS = "cleanslate/harness"
PEOPLE = tuple(f"P{i:02d}" for i in range(1, 11))
RETURNING = ("P04", "P10")
SITES = {
    "leonardo": ("AIFAC_P02_1636", "boost_usr_prod", "boost_usr_prod"),
    "galvani": ("gehler", "cpu-galvani,2080-galvani", "a100-galvani,2080-galvani"),
}


@dataclass(frozen=True)
class Node:
    key: str
    program: str
    args: list
    resources: Family
    deps: tuple = ()
    env: tuple = ()


def scopes(people, no_new, no_ret):
    folds = []
    if not no_new:
        folds += [("new", q, tuple(p for p in PEOPLE if p != q), (), "all") for q in people]
    if not no_ret:
        folds += [("ret", q, PEOPLE, (q,), "2,3") for q in people if q in RETURNING]
    return folds


def scope_args(people, first, data):
    args = ["--people", ",".join(people), "--data", str(data)]
    if first:
        args += ["--first-session-only", ",".join(first)]
    return args


def options(names):
    if not names:
        value = Variant.from_env()
        return {"selected": value}
    choices = {"stock": Variant(), "hard": Variant(hard=True), "cb": Variant(cb=True),
               "hardcb": Variant(True, True)}
    choices.update({f"{arm}-{ttt}": replace(variant, ttt=ttt) for arm, variant in tuple(choices.items())
                    for ttt in TTT_FORMS if ttt != "stock"})
    variants = names.split(",")
    if set(variants) - choices.keys() or len(variants) != len(set(variants)):
        raise ValueError(f"unknown or duplicate variants: {names}")
    return {name: choices[name] for name in variants}


class Plan:
    def __init__(self, run, data, variants):
        self.run, self.data, self.variants = run, data, variants
        self.nodes = {}
        self.manifests = {}

    def add(self, key, program, args, resources, deps=(), env=()):
        deps = tuple(deps)
        job = Node(key, program, args, resources, deps, tuple(env))
        if key in self.nodes and self.nodes[key] != job:
            raise ValueError(f"job key collision: {key}")
        if set(deps) - self.nodes.keys():
            raise ValueError(f"undefined prerequisite for {key}")
        self.nodes[key] = job
        return key

    def family(self, parent, family, seed, people, first, parts, *, base_only=False, dep=(), oof=None):
        args = ["--family", family, "--seed", str(seed), *scope_args(people, first, self.data),
                "--out", str(part_path(parts, family, seed))]
        if base_only:
            args += ["--base-only"]
        env = (("CAP250_ROUTE_OOF", str(oof)),) if oof else ()
        return self.add(f"{parent}.{family}.{seed}", f"{HERE}/worker.py", args,
                        FAMILIES[family], dep, env)

    def route(self, label, people, first):
        banks, streams = {}, []
        for held in people:
            fit, fit_first = subfold(people, first, held)
            key = "oof." + "-".join(fit) + ("." + "-".join(fit_first) if fit_first else "")
            root = self.run / "route" / key
            parts = root / "parts"
            required = [self.family(key, family, seed, fit, fit_first, parts, base_only=family == "geometry")
                        for family, seed in b2_fits()]
            query_first = held in first
            stream = f"{key}.query.{held}.{'first' if query_first else 'all'}"
            target = root / f"{held}_{'first' if query_first else 'all'}.npz"
            args = [*scope_args(fit, fit_first, self.data), "--parts", str(parts), "--query", held,
                    "--out", str(target)]
            if query_first:
                args += ["--query-first"]
            streams.append(self.add(stream, f"{HERE}/calibrate.py", args, Family(1, 25), required))
            banks[held] = str(target)
        manifest = self.run / "route" / f"{label}.json"
        self.manifests[manifest] = json.dumps(banks, indent=2) + "\n"
        return manifest, streams

    def fold(self, mode, held, people, first, sessions, orders, sealed, smoke):
        label = f"{mode}_{held}"
        parts = self.run / "parts" / label
        manifest_path, calibration = self.route(label, people, first)
        fits = {}
        for variant in self.variants.values():
            for family, seed in variant.fits():
                dep, oof = (calibration, manifest_path) if family == "geometry" else ((), None)
                fits[family, seed] = self.family(label, family, seed, people, first, parts, dep=dep, oof=oof)
        streams = {}
        for name, variant in self.variants.items():
            package = self.run / name / label / "pkg"
            assemble = self.add(f"{name}.{label}.assemble", f"{HERE}/assemble.py",
                                [*scope_args(people, first, self.data), "--parts", str(parts), "--out", str(package)],
                                Family(1, 20, False, memory="16G"),
                                [fits[task] for task in variant.fits()], variant.env().items())
            if mode == "all":
                reports = package.parent / "parity"
                decisions = [
                    self.add(f"{name}.{label}.parity.{person}", f"{HERE}/verify.py",
                             ["--pkg", str(package), "--sealed", str(sealed), "--data", str(self.data),
                              "--person", person, "--shuffle", orders.split(",")[0], "--report-only",
                              "--out", str(reports / f"{person}.json")],
                             Family(1, 25), [assemble])
                    for person in PEOPLE
                ]
                streams[name] = self.add(f"{name}.{label}.parity", f"{HERE}/parity_report.py",
                                         ["--input", str(reports), "--out", str(package.parent / "parity.json")],
                                         Family(1, 10, False, cpus=2, memory="8G"), decisions)
            else:
                output = f"{'smoke' if smoke else 'pred'}_{label}.npz"
                streams[name] = self.add(f"{name}.{label}.stream", f"{HERE}/stream_fold.py",
                                         ["--pkg", str(package), "--held", held, "--sessions", sessions,
                                          "--orders", orders, "--data", str(self.data), "--out",
                                          str(self.run / name / output)],
                                         Family(1, 30), [assemble])
        return streams

    def build(self, folds, orders, sealed, smoke):
        streams = {name: [] for name in self.variants}
        for fold in folds:
            for name, job in self.fold(*fold, orders, sealed, smoke).items():
                streams[name].append(job)
        if folds[0][0] == "all" or smoke:
            return
        table = "".join("\t".join((mode, held, ",".join(train), ",".join(first) or "-", sessions)) + "\n"
                        for mode, held, train, first, sessions in folds)
        for name, jobs in streams.items():
            root = self.run / name
            self.manifests[root / "jobs.tsv"] = table
            self.add(f"{name}.score", f"{HARNESS}/score.py", [str(root), "--data", str(self.data)],
                     Family(1, 20, False, cpus=2, memory="8G"), jobs)

    def write(self):
        self.run.mkdir(parents=True, exist_ok=False)
        (self.run / "plan.json").write_text(self.describe())
        for path, content in self.manifests.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    def describe(self):
        return json.dumps([asdict(node) for node in self.nodes.values()], indent=2) + "\n"


def submit(plan, args):
    # Arrays keep family×seed tasks individually addressable while reducing sbatch calls.
    from scheduler import Slurm

    scripts = {node.program for node in plan.nodes.values()}
    scripts.update(f"{HERE}/families/{node.args[1]}.py" for node in plan.nodes.values() if node.program.endswith("/worker.py"))
    missing = sorted(s for s in scripts if not (args.release / s).is_file())
    if missing:
        raise FileNotFoundError("release is missing " + ", ".join(missing))
    plan.write()
    Slurm(args).submit(plan.nodes.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--release", type=Path, default=ROOT)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--people", default=",".join(PEOPLE))
    parser.add_argument("--orders", default="71000,71001,71002,71003")
    parser.add_argument("--variants", help="stock,hardcb etc. Default reads CAP250_HARD, CAP250_CB and CAP250_TTT")
    parser.add_argument("--no-new", action="store_true")
    parser.add_argument("--no-ret", action="store_true")
    parser.add_argument("--all-ten", action="store_true")
    parser.add_argument("--sealed", type=Path)
    parser.add_argument("--smoke", action="store_true", help="reduced family fits; scores and archive comparison disabled")
    parser.add_argument("--submit", action="store_true", help="default only prints the graph")
    parser.add_argument("--afterok", type=int, help="same-cluster job that must pass before this graph starts")
    parser.add_argument("--site", choices=SITES, default="leonardo")
    parser.add_argument("--account")
    parser.add_argument("--cpu-partition")
    parser.add_argument("--gpu-partition")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--exclude", default="")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--python", type=Path, help="absolute Python path if not using an Apptainer image")
    parser.add_argument("--revision", default=os.environ.get("CAP250_REV"))
    args = parser.parse_args()
    args.release, args.run = args.release.resolve(), args.run.resolve()
    account, cpu, gpu = SITES[args.site]
    args.account, args.cpu_partition, args.gpu_partition = (
        args.account or account, args.cpu_partition or cpu, args.gpu_partition or gpu)
    work = os.environ.get("WORK") if args.site == "leonardo" else None
    args.assets = (args.assets or (Path(work) / "ni26" if work else args.release)).resolve()
    args.data = (args.data or args.assets / "data/proteus_windows.npz").resolve()
    if args.submit and args.image is None and args.python is None and args.site == "leonardo" and work:
        args.python = Path(work) / "ni26/.venv312/bin/python"
    if args.submit:
        runner = args.image or args.python
        if runner is not None and not runner.is_file():
            parser.error(f"worker runtime does not exist: {runner}")
    order_ids = args.orders.split(",")
    people = args.people.split(",")
    if (not all(order.isdecimal() for order in order_ids) or len(set(order_ids)) != len(order_ids)
            or set(people) - set(PEOPLE) or len(set(people)) != len(people)):
        parser.error("invalid or repeated person/order")
    if args.afterok is not None and args.afterok <= 0:
        parser.error("--afterok requires a positive Slurm job ID")
    if args.submit and ((args.image is None) == (args.python is None) or not args.data.is_file()):
        parser.error("submission requires an existing data file and exactly one of --image or --python")
    if args.all_ten and (args.smoke or args.sealed is None):
        parser.error("--all-ten requires a fresh full fit and --sealed")
    if args.all_ten:
        args.sealed = args.sealed.resolve()
    folds = [("all", "ten", PEOPLE, (), "all")] if args.all_ten else scopes(people, args.no_new, args.no_ret)
    if not folds:
        parser.error("no folds selected")
    plan = Plan(args.run, args.data, options(args.variants))
    plan.build(folds, args.orders, args.sealed, args.smoke)
    if args.submit:
        submit(plan, args)
    else:
        sys.stdout.write(plan.describe())


if __name__ == "__main__":
    main()
