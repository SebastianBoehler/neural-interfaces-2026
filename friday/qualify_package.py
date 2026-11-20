"""Friday: build and qualify a candidate package ZIP (no upload, no Codabench contact).

  python friday/qualify_package.py local   CAND OUT [--expect-sha H] [--expect-score S] [--skip-replay]
  python friday/qualify_package.py cluster OUT [--ref-zip Z] [--submit]   # worker-image replays, A100 + RTX 2080 Ti
  python friday/qualify_package.py pair CAND OUT --ref-zip Z --expect-sha H  # Leonardo A100 Slurm
  python friday/qualify_package.py report  OUT [--fetch]                  # timing projection + PASS/FAIL line

CAND is a package folder (submission.py + data files, flat) or a finished ZIP. A folder is zipped
deterministically (sorted names, fixed timestamps, deflate); a ZIP is used byte for byte.
`local` writes OUT/package.zip and OUT/local.json: SHA-256, size, members, CRC, layout, official CPU
replay (scripts/replay_package.py: ingestion + scoring, files unchanged), reset/prefix/future checks.
`cluster` writes OUT/cluster/{a100,rtx}.sbatch (replays ref, cand, ref, cand inside the pinned worker SIF)
and with --submit copies the ZIPs and submits via ssh mlcloud-galvani (eduVPN on). `report` reads
OUT/cluster results and projects the sealed time as SEALED_REF_S * cand_wall / ref_wall on the same device
(min over the repeats). The ratio is the only transferable number: the sealed worker, session count and
RTX model are unverified (evidence/wave_19nov/laneF_package.md). The last line is PASS, FAIL or PENDING.
"""
import argparse, hashlib, json, os, re, subprocess, sys, tempfile, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF_ZIP = ROOT / "runs/sealed/final_zips/v66_cp1_cap250.zip"
REF_SHA = "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed"
REF_SCORE = 0.9945788530      # docs/V6_BUILD.md: Mac CPU replay of the exact cap250 ZIP, public windows
SEALED_REF_S = 2554.9166      # sealed receipt of cap250 (submission 1058969)
LIMIT_S = 3600.0              # official sealed budget, 60 min on one RTX PRO 4500 (upstream main 8447e181, 19 Nov)
MARGIN = 0.85                 # SELF-IMPOSED safety margin, not an official rule: projected time must stay below MARGIN * LIMIT_S
MAX_MB = float(os.environ.get("NI26_MAX_MB", 90))  # SELF-IMPOSED, not an official limit (no per-ZIP cap; 15 GB profile storage, check free space live); cap250 is 66.8 MB; final_stack 90.5 MB
ALLOWED = {".py", ".npz", ".pt", ".so"}
REMOTE_REF = "/mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117/runs/sealed/final_zips/v66_cp1_cap250.zip"
REMOTE = "/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026"
HOST = "mlcloud-galvani"
TOL = 5e-7


def sha(b):
    return hashlib.sha256(b).hexdigest()


def build(cand, out):
    z = out / "package.zip"
    if cand.suffix == ".zip":
        z.write_bytes(cand.read_bytes())
        return z
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(cand.iterdir()):
            if p.is_file():
                zf.writestr(zipfile.ZipInfo(p.name, (1980, 1, 1, 0, 0, 0)), p.read_bytes(), zipfile.ZIP_DEFLATED)
    return z


def inspect(z):
    b = z.read_bytes()
    with zipfile.ZipFile(z) as zf:
        bad = zf.testzip()
        infos = zf.infolist()
        names = [i.filename for i in infos]
        layout = []
        if "submission.py" not in names:
            layout.append("no root submission.py")
        layout += [f"directory/nested entry {n}" for n in names if n.endswith("/") or "/" in n]
        layout += [f"extra .py {n}" for n in names if n.endswith(".py") and n != "submission.py"]
        layout += [f"unsupported suffix {n}" for n in names if Path(n).suffix not in ALLOWED]
        layout += ["duplicate names"] if len(set(names)) != len(names) else []
        if len(b) > MAX_MB * 2**20:
            layout.append(f"ZIP {len(b) / 2**20:.1f} MB above {MAX_MB} MB storage allowance")
        files = {i.filename: sha(zf.read(i)) for i in infos}
    return {"sha256": sha(b), "bytes": len(b), "n_members": len(names), "crc_bad_member": bad,
            "layout_problems": layout, "members_sha256": files,
            "manifest_sha256": sha(json.dumps(files, sort_keys=True).encode())}


def replay(z, out, device="cpu", name=None):
    r = out / (name or f"replay_{device}")
    subprocess.run([sys.executable, str(ROOT / "scripts/replay_package.py"), str(z), str(r), device],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
                   env={**os.environ, "BENCHOPT_DATA_HOME": str(ROOT / "data"), "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"})  # finds benchopt
    j = json.loads((r / "replay.json").read_text())
    return {"score": j["scores"], "files_unchanged": j["files_unchanged"], "wall_s": j["ingestion_wall_seconds"],
            "zip_sha256": j["zip_sha256"]}


def cmd_pair(a):
    """Qualify two warm-up replays per arm on the A100 allocated by Slurm."""
    import torch

    if not os.environ.get("SLURM_JOB_ID") or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("expected a Slurm A100 allocation")
    inputs = {"ref": (a.ref_zip, REF_SHA), "cand": (a.cand, a.expect_sha)}
    for arm, (archive, expected) in inputs.items():
        manifest = inspect(archive)
        if manifest["sha256"] != expected or manifest["crc_bad_member"] or manifest["layout_problems"]:
            raise ValueError(f"{arm}: archive identity, CRC or layout failed")
    a.out.mkdir(parents=True, exist_ok=False)
    walls = {"ref": [], "cand": []}
    for index, arm in enumerate(("ref", "cand", "ref", "cand"), 1):
        name = f"a100_{index}_{arm}"
        result = replay(inputs[arm][0], a.out, "cuda", name)
        log = (a.out / name / "ingestion.log").read_text()
        errors = [int(n) for n in re.findall(r"\[v6\].* errors (\d+)", log)]
        if not errors or any(errors) or not result["files_unchanged"]:
            raise RuntimeError(f"{name}: missing or degraded v6 stream")
        walls[arm].append({**result, "v6_streams": len(errors)})
        print(name, result["wall_s"], result["score"], flush=True)

    def score(replay_result):
        return {k: v for k, v in replay_result["score"].items() if k != "duration"}

    stable = all(score(first) == score(second) for first, second in walls.values())
    reference = score(walls["ref"][0])["balanced_accuracy_cell_mean"]
    candidate = score(walls["cand"][0])["balanced_accuracy_cell_mean"]
    passed = stable and abs(reference - REF_SCORE) < TOL and abs(candidate - reference) <= .005
    report = {"gpu": torch.cuda.get_device_name(0), "job": os.environ["SLURM_JOB_ID"], "runs": walls,
              "repeat_scores_equal": stable, "replay_pass": passed}
    (a.out / "pair.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if passed else 1


def causal_checks(z, n=45, k=30):
    """Reset, prefix and future-perturbation checks on real public windows, official fresh_model path."""
    import numpy as np, torch
    sys.path[:0] = [str(ROOT / "2026-competition"), str(ROOT / "native/tools")]
    from benchmark_utils.streaming import fresh_model
    from runtime_reference import load_solver
    torch.set_num_threads(1)
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as d:
        subj, sess = d["subject"], d["session"]
        def run_of(s, e):
            i = np.flatnonzero((subj == s) & (sess == e))[:n]
            return torch.from_numpy(d["X"][i].copy())
        a, b = run_of("Dreyer2026Proteus/P01", "1"), run_of("Dreyer2026Proteus/P02", "1")
    with tempfile.TemporaryDirectory() as t, zipfile.ZipFile(z) as zf:
        zf.extractall(t)
        tpl = load_solver(Path(t), "cpu")

        def stream(x):
            m = fresh_model(tpl)
            return [int(m.predict(x[i:i + 1])[0]) for i in range(len(x))]
        base = stream(a)
        res = {"n_windows": len(a), "k": k}
        stream(b)                                   # another session between two runs of the same session
        res["reset_repeat_equal"] = stream(a) == base
        res["prefix_equal"] = stream(a[:k]) == base[:k]
        fut = a.clone()
        fut[k:] += 5.0 * torch.randn_like(fut[k:])
        res["future_perturbation_equal"] = stream(fut)[:k] == base[:k]
        res["labels_valid"] = set(base) <= {0, 1, 2} and len(set(base)) >= 1
    res["pass"] = all(v for kk, v in res.items() if kk.endswith(("equal", "valid")))
    return res


def cmd_local(a):
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    z = build(a.cand.resolve(), out)
    r = {"inspect": inspect(z), "cand": str(a.cand)}
    r["causal"] = causal_checks(z)
    if not a.skip_replay:
        r["cpu_replay"] = replay(z, out)
    i = r["inspect"]
    g = {"sha_matches": a.expect_sha is None or i["sha256"] == a.expect_sha,
         "crc_ok": i["crc_bad_member"] is None, "layout_ok": not i["layout_problems"],
         "causal_ok": r["causal"]["pass"]}
    if not a.skip_replay:
        s = r["cpu_replay"]["score"].get("balanced_accuracy_cell_mean")
        g["replay_files_unchanged"] = r["cpu_replay"]["files_unchanged"]
        if a.expect_score is not None:
            g["replay_score_matches"] = s is not None and abs(s - a.expect_score) < TOL
    r["gates"], r["pass_local"] = g, all(g.values())
    (out / "local.json").write_text(json.dumps(r, indent=1) + "\n")
    print(f"{i['sha256']}  {i['bytes']} B  {i['n_members']} members")
    print(json.dumps({"causal": r["causal"], "gates": g}, indent=1))
    print("LOCAL " + ("PASS" if r["pass_local"] else "FAIL"))
    return 0 if r["pass_local"] else 1


SBATCH = """#!/bin/bash
#SBATCH -A gehler
#SBATCH -p {part}
#SBATCH --gres=gpu:{gres}:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t {time}
#SBATCH -J ni26-qual-{tag}
#SBATCH -o {remote}/{tag}-%j.out
set -euo pipefail
cd {repo}
SIF=runs/cloud/ni26_worker.sif; test -f $SIF
cd {remote}
echo "{cand_sha}  cand.zip" | sha256sum -c -; echo "{ref_sha}  ref.zip" | sha256sum -c -
i=0
for arm in ref cand ref cand; do
  n=$((++i)); out={remote}/{tag}_${{n}}_$arm
  apptainer exec --nv --bind {repo} --bind {remote} --env BENCHOPT_DATA_HOME={repo}/data {repo}/$SIF \\
    bash -c "cd {repo} && python scripts/replay_package.py {remote}/$arm.zip $out cuda"
done
"""


def cmd_cluster(a):
    out = a.out.resolve()
    loc = json.loads((out / "local.json").read_text())
    ref = a.ref_zip.resolve()
    if ref != REF_ZIP:
        sys.exit("a custom --ref-zip needs its own upload line; the default reference is already on Galvani")
    remote = f"{REMOTE}/qual_{loc['inspect']['sha256'][:12]}"
    d = out / "cluster"
    d.mkdir(exist_ok=True)
    for tag, part, gres, tm in (("a100", "a100-galvani", "a100", "01:30:00"),
                                ("rtx", "2080-galvani", "rtx2080ti", "03:00:00")):
        (d / f"{tag}.sbatch").write_text(SBATCH.format(
            part=part, gres=gres, time=tm, tag=tag, remote=remote, repo=REMOTE,
            cand_sha=loc["inspect"]["sha256"], ref_sha=sha(ref.read_bytes())))
    cmds = [f"ssh {HOST} mkdir -p {remote}", f"rsync -t --partial {out / 'package.zip'} {HOST}:{remote}/cand.zip",
            f"ssh {HOST} ln -sf {REMOTE_REF} {remote}/ref.zip"] + [f"scp {d / t}.sbatch {HOST}:{remote}/" for t in ("a100", "rtx")]
    cmds += [f"ssh {HOST} 'cd {remote} && sbatch --parsable {t}.sbatch'" for t in ("a100", "rtx")]
    (d / "remote").write_text(remote + "\n")
    print("\n".join(cmds))
    if a.submit:
        for c in cmds:
            subprocess.run(c, shell=True, check=True)
    return 0


def walls(d, tag):
    """Per-arm ingestion wall seconds of the repeats in a result dir."""
    w = {"ref": [], "cand": []}
    for p in sorted(d.glob(f"{tag}_*_*/replay.json")):
        j = json.loads(p.read_text())
        w[p.parent.name.rsplit("_", 1)[1]].append((j["ingestion_wall_seconds"], j["scores"], j["files_unchanged"]))
    return w


def cmd_report(a):
    out = a.out.resolve()
    d = out / "cluster"
    rem = (d / "remote").read_text().strip()
    if a.fetch:
        subprocess.run(f"scp -r '{HOST}:{rem}/*_*_*' {d}/", shell=True, check=True)
    loc = json.loads((out / "local.json").read_text())
    res, ok, pending = {}, True, False
    for tag in ("a100", "rtx"):
        w = walls(d, tag)
        if len(w["ref"]) < 2 or len(w["cand"]) < 2:
            res[tag], pending = "pending", True
            continue
        rt, ct = min(x[0] for x in w["ref"]), min(x[0] for x in w["cand"])
        proj = SEALED_REF_S * ct / rt
        same = all(x[1] == w["cand"][0][1] for x in w["cand"]) and all(x[2] for x in w["cand"] + w["ref"])
        res[tag] = {"ref_wall_s": rt, "cand_wall_s": ct, "ratio": round(ct / rt, 4), "projected_sealed_s": round(proj, 1),
                    "within_margin": proj <= MARGIN * LIMIT_S, "scores_cand": w["cand"][0][1],
                    "repeat_scores_identical_files_unchanged": same}
        ok &= res[tag]["within_margin"] and same
    (out / "report.json").write_text(json.dumps({"local": loc["gates"], "timing": res}, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    ok &= loc["pass_local"]
    print("QUALIFY " + ("FAIL" if not ok else "PENDING-CLUSTER" if pending else "PASS"))
    return 0 if ok and not pending else 1


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    l = s.add_parser("local")
    l.add_argument("cand", type=Path); l.add_argument("out", type=Path)
    l.add_argument("--expect-sha"); l.add_argument("--expect-score", type=float); l.add_argument("--skip-replay", action="store_true")
    c = s.add_parser("cluster")
    c.add_argument("out", type=Path); c.add_argument("--ref-zip", type=Path, default=REF_ZIP); c.add_argument("--submit", action="store_true")
    q = s.add_parser("pair")
    q.add_argument("cand", type=Path); q.add_argument("out", type=Path)
    q.add_argument("--ref-zip", type=Path, required=True); q.add_argument("--expect-sha", required=True)
    r = s.add_parser("report")
    r.add_argument("out", type=Path); r.add_argument("--fetch", action="store_true")
    a = p.parse_args()
    sys.exit({"local": cmd_local, "cluster": cmd_cluster, "pair": cmd_pair, "report": cmd_report}[a.cmd](a))


if __name__ == "__main__":
    main()
