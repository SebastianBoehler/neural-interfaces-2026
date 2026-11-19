"""Submit the clean-slate LOPO + returning folds of one team (Slurm only; stdlib only, run on the Galvani login node).

python run_lopo.py --team example --orders 71000,71001 [--people P01,P04] [--tag T] [--no-ret] [--no-new]
                   [--partition 2080-galvani --gres gpu:rtx2080ti:1] [--time 06:00:00] [--env NAME=VAL,...] [--exclude NODES] [--resume] [--dry]
Per held person P in --people (default P01-P10):
  new : train.py --people <other nine>; stream ALL sessions of P            (held-new, 10 people)
  ret : P04 and P10 only: train.py --people <all ten> --first-session-only P; stream sessions 2,3 of P   (returning)
Each fold is one array task on a GPU: train then stream (real Solver, one window per predict, deep copy + reset per
session, full-session shuffle per order seed). A dependent CPU job runs score.py -> RUN/cells.npz, scores.json.
Release dir (CS_REL, default parent of cleanslate/) must hold the rsynced cleanslate/ tree.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PEOPLE = [f"P{k:02d}" for k in range(1, 11)]
RETURNING = ("P04", "P10")
HERE = Path(__file__).resolve().parent
BASE = "/mnt/lustre/work/gehler/gfh527"


def jobs(people, new, ret):
    rows = []
    if new:
        rows += [("new", p, ",".join(q for q in PEOPLE if q != p), "-", "all") for p in people]
    if ret:
        rows += [("ret", p, ",".join(PEOPLE), p, "2,3") for p in people if p in RETURNING]
    return rows


def sh(cmd, dry):
    print(" ".join(cmd))
    if dry:
        return "DRY"
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--orders", default="71000,71001")
    ap.add_argument("--people", default=",".join(PEOPLE))
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-new", action="store_true")
    ap.add_argument("--no-ret", action="store_true")
    ap.add_argument("--partition", default="2080-galvani")
    ap.add_argument("--gres", default="gpu:rtx2080ti:1")
    ap.add_argument("--time", default="06:00:00")
    ap.add_argument("--env", default="", help="NAME=VAL,... passed to train.py and the stream (e.g. ablation switches)")
    ap.add_argument("--exclude", default="galvani-cn122", help="nodes without a working GPU context (CUDA_ERROR_UNKNOWN)")
    ap.add_argument("--resume", action="store_true", help="re-run only the folds of an existing --tag run lacking predictions")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    people = a.people.split(",")
    assert set(people) <= set(PEOPLE) and all(o.isdigit() for o in a.orders.split(","))
    rel = Path(os.environ.get("CS_REL", HERE.parent.parent)).resolve()
    assert (rel / "cleanslate" / a.team / "train.py").exists(), f"missing cleanslate/{a.team}/train.py in {rel}"
    tag = a.tag or "o" + a.orders.replace(",", "_") + ("" if not a.env else "_" + a.env.replace("=", "").replace(",", "_"))
    run = rel / "runs" / a.team / tag
    table = jobs(people, not a.no_new, not a.no_ret)
    assert table, "nothing to run"
    todo = list(range(len(table)))
    if a.resume:
        assert (run / "jobs.tsv").exists(), f"nothing to resume in {run}"
        table = [tuple(l.split("\t")) for l in (run / "jobs.tsv").read_text().splitlines()]
        todo = [i for i, r in enumerate(table) if not (run / f"pred_{r[0]}_{r[1]}.npz").exists()]
        assert todo, "all folds have predictions"
    elif not a.dry:
        assert not (run / "jobs.tsv").exists(), f"{run} exists; pick another --tag (or --resume)"
        run.mkdir(parents=True)
        (run / "jobs.tsv").write_text("".join("\t".join(r) + "\n" for r in table))
    env = f"ALL,CS_REL={rel},CS_TEAM={a.team},CS_RUN={run},CS_ORDERS={a.orders.replace(',', ';')}"
    env_cs = a.env.replace(",", ";")  # sbatch --export splits on commas; fold.sbatch re-joins below
    cmd = ["sbatch", "--parsable", f"--array={','.join(map(str, todo))}", f"--partition={a.partition}", f"--exclude={a.exclude}", f"--gres={a.gres}",
           f"--time={a.time}", f"--chdir={run}", f"--export={env},CS_ENV_SEMI={env_cs}", str(HERE / "fold.sbatch")]
    jid = sh(cmd, a.dry)
    sh(["sbatch", "--parsable", f"--dependency=afterok:{jid}", f"--chdir={run}", f"--export=ALL,CS_REL={rel},CS_RUN={run}",
        str(HERE / "score.sbatch")], a.dry)
    print(f"run dir {run}; folds {len(table)}; scores.json lands there after the dependent score job")


if __name__ == "__main__":
    sys.exit(main())
