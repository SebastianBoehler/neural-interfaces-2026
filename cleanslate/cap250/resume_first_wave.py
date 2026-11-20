"""Resubmit unsuccessful first-wave tasks and restart the saved Slurm graph."""

import argparse
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from components import Family
from graph import Node
from scheduler import Slurm
from waves import dispatch_job


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def terminal_states(job_ids):
    bases = sorted({job.split("_")[0] for job in job_ids})
    output = subprocess.check_output(
        ["sacct", "-X", "-nP", "-j", ",".join(bases), "-o", "JobID,State%40,ExitCode"],
        text=True,
    )
    states = {}
    for line in output.splitlines():
        job, state, exit_code = line.split("|")
        if job in job_ids:
            if state == "COMPLETED" and exit_code != "0:0":
                raise RuntimeError(f"nonzero exit code for completed job {job}: {exit_code}")
            states[job] = state.split()[0]
    if states.keys() != job_ids:
        raise RuntimeError(f"missing Slurm accounting records: {sorted(job_ids - states.keys())}")
    return states


def resume(args):
    run = args.run
    params = json.loads((run / "dispatch.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    boundary = params["wave_size"]
    if not 0 < args.batch_size <= boundary or boundary % args.batch_size:
        raise ValueError("batch size must divide the original wave size")
    first = {row["key"]: row for row in plan[:boundary]}
    dispatch = read_rows(run / "dispatch.jsonl")
    previous = read_rows(run / "jobs.jsonl")
    jobs = {row["node"]: row["job"] for row in previous}
    if [row["offset"] for row in dispatch] != [0, boundary] or jobs.keys() != first.keys() or len(previous) != len(first):
        raise RuntimeError("expected one submitted wave and its original second-wave dispatcher")

    old_dispatch = dispatch[1]["job"]
    states = terminal_states({*jobs.values(), old_dispatch, dispatch[0]["job"]})
    if states[old_dispatch] != "CANCELLED" or states[dispatch[0]["job"]] != "COMPLETED":
        raise RuntimeError("original launch must be complete and its successor cancelled")
    if set(states.values()) - {"COMPLETED", "FAILED", "CANCELLED"}:
        raise RuntimeError("all original wave tasks must have a terminal accounting state")
    completed = {key for key, job in jobs.items() if states[job] == "COMPLETED"}
    if any(set(first[key]["deps"]) - completed for key in completed):
        raise RuntimeError("a completed task has an unsuccessful prerequisite")
    retry = [
        Node(key, row["program"], row["args"], Family(**row["resources"]),
             tuple(dep for dep in row["deps"] if dep not in completed),
             tuple(tuple(pair) for pair in row["env"]))
        for key, row in first.items() if key not in completed
    ]
    if not retry:
        raise RuntimeError("no unsuccessful first-wave tasks to resubmit")
    print(f"Keeping {len(completed)} completed jobs. Retrying {len(retry)} tasks:", flush=True)
    for node in retry:
        print(f"  {jobs[node.key]} {node.key} after {', '.join(node.deps) or 'verified prerequisites'}", flush=True)
    if not args.submit:
        return

    params.update(release=str(args.release), revision=args.revision, cb_stagea=str(args.stage_a),
                  wave_size=args.batch_size, afterok=None)
    config = run / "dispatch-recovery.json"
    for seed in range(3):
        receipt = args.stage_a / f"external_C_{seed}.json"
        if json.loads(receipt.read_text()).get("backbone") != "cbramod" or not receipt.with_suffix(".pt").is_file():
            raise ValueError(f"missing CBraMod checkpoint or provenance: {receipt}")
    runtime = dict(params)
    for key in ("run", "release"):
        runtime[key] = Path(params[key])
    if Path(params["run"]) != run or not (args.release / "cleanslate/cap250/slurm.sh").is_file():
        raise ValueError("run path mismatch or missing staged release")
    with config.open("x") as handle:
        handle.write(json.dumps(params, indent=2) + "\n")
    for key, value in (("CAP250_CB_CHECKPOINT", params["cb_checkpoint"]),
                       ("CAP250_CB_STAGEA", params["cb_stagea"]),
                       ("CAP250_HARD_SOURCE", params["hard_source"])):
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    scheduler = Slurm(SimpleNamespace(**runtime))
    scheduler.submit(retry, batch=0,
                     before_release=lambda parents: dispatch_job(scheduler, config, boundary, parents))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--submit", action="store_true")
    resume(parser.parse_args())
