"""Slurm dispatchers for a DAG larger than the site's pending-job allowance.

A dispatcher starts only after the previous wave completed. Each job it queues
waits for the dispatcher and its own in-wave prerequisites by afterok. The next
dispatcher waits for every base job in this wave, including all array tasks.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

from components import Family
from graph import Node
from scheduler import Slurm


def dispatch_job(scheduler, config, offset, parents):
    body = ["bash", scheduler.runner, "cleanslate/cap250/waves.py", str(config), str(offset)]
    job = scheduler.sbatch(f"cap250-dispatch-{offset}", Family(1, 30, False, cpus=1, memory="3G"),
                           parents, ["--wrap=" + shlex.join(body)])
    with (scheduler.run / "dispatch.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(offset=offset, job=job, parents=parents)) + "\n")
    print(f"dispatcher {job} offset {offset} afterok {':'.join(parents)}", flush=True)
    return job


def queue(args):
    config = args.run / "dispatch.json"
    args.revision = args.revision or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.release, text=True).strip()
    params = {**vars(args), "cb_checkpoint": os.environ.get("CAP250_CB_CHECKPOINT"),
              "cb_stagea": os.environ.get("CAP250_CB_STAGEA"),
              "hard_source": os.environ.get("CAP250_HARD_SOURCE")}
    config.write_text(json.dumps(params, default=str, indent=2) + "\n")
    scheduler = Slurm(args)
    dispatch_job(scheduler, config, 0, (str(args.afterok),) if args.afterok else ())
    scheduler.release()


def run(config, offset):
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("cap250 dispatchers must run as Slurm compute jobs")
    args = SimpleNamespace(**json.loads(config.read_text()))
    for key in ("assets", "data", "image", "python", "release", "run"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, Path(value))
    for key, value in (("CAP250_CB_CHECKPOINT", args.cb_checkpoint), ("CAP250_CB_STAGEA", args.cb_stagea),
                       ("CAP250_HARD_SOURCE", args.hard_source)):
        if value is not None:
            os.environ[key] = value
    args.afterok = int(job_id)
    plan = json.loads((args.run / "plan.json").read_text())
    if offset % args.wave_size or not 0 <= offset < len(plan):
        raise ValueError(f"invalid cap250 wave offset: {offset}")
    end = min(offset + args.wave_size, len(plan))
    completed = {node["key"] for node in plan[:offset]}
    nodes = [
        Node(row["key"], row["program"], row["args"], Family(**row["resources"]),
             tuple(key for key in row["deps"] if key not in completed),
             tuple(tuple(pair) for pair in row["env"]))
        for row in plan[offset:end]
    ]
    scheduler = Slurm(args)

    def continuation(base_ids):
        if end < len(plan):
            dispatch_job(scheduler, config, end, base_ids)

    try:
        scheduler.submit(nodes, batch=offset // args.wave_size + 1, before_release=continuation)
    except (RuntimeError, subprocess.CalledProcessError):
        if scheduler.base_ids:
            subprocess.run(["scancel", *scheduler.base_ids], env=scheduler.env, check=False)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("offset", type=int)
    params = parser.parse_args()
    run(params.config, params.offset)
