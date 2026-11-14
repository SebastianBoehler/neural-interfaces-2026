"""Assemble a v6 package: the v6 bundle, v5.1b's data files unchanged, the eye-net and rest-net traces and the v6 keys.

Usage: python lean/tools/build6.py V51B_DIR MEMBERS_DIR OUT_DIR [--steps 5 --every 10 --t0 30 --lr 5e-4] [--ft-off]
       [--v61 --labels v6r]   (v6.1: the lean/src61 bundle; fine-tuning labels from the v6 chain without fine-tuning)
       [--v62 --routed 0|1|2] (v6.2: the lean/src62 bundle; 0 = v6.1, 1 = v6.2-safe, 2 = v6.2)
       [--v62g --cap 400 --budget 3000] (v6.2g: lean/src62g, fine-tuning cost guards)
       [--v62i --cap 400 --session-s S] (v6.2i: lean/src62i, per-session limit instead of the process budget)
       [--v62h ...]           (v6.2h: lean/src62h, v6.2g plus the step62 exception guard; v6.2g's keys)
MEMBERS_DIR holds eye_s<k>.pt, rest_s<k>.pt and rest_s<k>.npz (scripts/train_v6_members.py). Keys added to weights.npz:
  v6_eye_files, v6_rest_files, v6_rest_{align,k,G,scale,shrink} (equal across rest seeds: asserted), v6_rest_w = 1/16
  (weight 1 against the replay's 15 units of the v5.1b mix), v6_slots (fbraw, fb, sinc, fbl = replay slots 5, 6, 7, 8;
  fb at k 5, slot 9, stays frozen), v6_slot_<s> (member indices), v6_ft_member (seed 0 of each), v6_ft_t0 / every /
  steps / lr / lam / seed (gC5: 30 / 10 / 5 / 5e-4 / 1 / 0). Every v5.1b array is unchanged (asserted).
The v6 bundle must be current; the package must load. OUT_DIR.sha256 lists the package files' digests."""
import hashlib, re, shutil, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
V62I = "--v62i" in sys.argv  # v6.2i: lean/src62i, v6_ft_session_s instead of v6_ft_budget_s
V62H = "--v62h" in sys.argv or V62I  # v6.2h: lean/src62h (v6.2g plus the step62 exception guard), same keys as v6.2g
V62G = "--v62g" in sys.argv or V62H
V62 = "--v62" in sys.argv or V62G
V61 = "--v61" in sys.argv or V62
if not V61:
    sys.argv.append("--v6")
import bundle as B  # noqa: E402  (--v6: lean/src6 -> lean_v6; --v61: lean/src61 -> lean_v61; bundle removes the flag)
from record import load_model  # noqa: E402

SLOTS = {"fbraw": r"sbn_fbcnet_raw_k20_s(\d)\.pt", "fb": r"sbn_k2_s(\d)\.pt", "sinc": r"sbn_sincshallow_ea_ptm7r_rot05_k20_s(\d)\.pt",
         "fbl": r"sbn_fblight_ea_rot05_k5_s(\d)\.pt"}


def arg(k, d):
    return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d


def main():
    src, mem, out = (Path(a).resolve() for a in sys.argv[1:4])
    assert B.OUT.read_text() == B.bundle(), f"{B.OUT} is stale: run lean/tools/bundle.py {'--v61' if V61 else '--v6'}"
    labels = {"v51b": 0, "v6r": 1}[arg("--labels", "v51b")]
    assert labels == 0 or V61, "--labels v6r needs the v6.1 bundle (--v61)"
    routed = int(arg("--routed", "0")); assert routed == 0 or V62, "--routed needs the v6.2 bundle (--v62)"
    assert not out.exists(), f"{out} exists"
    assert not V62I or "--session-s" in sys.argv, "--v62i needs --session-s (the per-session fine-tuning limit)"
    z = dict(np.load(src / "weights.npz", allow_pickle=False))
    eye, rest = sorted(p.name for p in mem.glob("eye_s*.pt")), sorted(p.name for p in mem.glob("rest_s*.pt"))
    assert eye and rest, f"no eye or rest traces in {mem}"
    R = [dict(np.load(mem / r.replace(".pt", ".npz"))) for r in rest]
    for k in ("align", "k", "G", "scale", "shrink"):
        assert all(np.array_equal(r[k], R[0][k]) for r in R), f"rest seeds differ in their input stream key {k}"
    files = {i: str(z[f"net{i}_sjit_file"]) for i in range(int(z["n_nets"])) if f"net{i}_sjit_file" in z}
    slot = {s: sorted((int(re.fullmatch(p, f).group(1)), i) for i, f in files.items() if re.fullmatch(p, f)) for s, p in SLOTS.items()}
    assert all(len(v) == 6 and v[0][0] == 0 for v in slot.values()), slot
    new = dict(z, v6_eye_files=np.array(eye), v6_rest_files=np.array(rest), v6_rest_w=np.float64(1 / 16), v6_slots=np.array(list(SLOTS)),
               v6_ft_member=np.array([slot[s][0][1] for s in SLOTS], np.int64), v6_ft_t0=np.int64(arg("--t0", 30)),
               v6_ft_every=np.int64(arg("--every", 10)), v6_ft_steps=np.int64(arg("--steps", 5)), v6_ft_lr=np.float64(arg("--lr", 5e-4)),
               v6_ft_lam=np.float64(1.0), v6_ft_seed=np.int64(0), **({"v6_ft_labels": np.int64(labels)} if V61 else {}), **({"v6_routed": np.int64(routed)} if V62 else {}),
               **({"v6_ft_cap": np.int64(arg("--cap", 400)), "v6_ft_session_s": np.float64(arg("--session-s", 0.0))} if V62I else
                 {"v6_ft_cap": np.int64(arg("--cap", 400)), "v6_ft_budget_s": np.float64(arg("--budget", 3000.0))} if V62G else {}), v6_ft_on=np.int64(0 if "--ft-off" in sys.argv else 1),
               **{f"v6_slot_{s}": np.array([i for _, i in v], np.int64) for s, v in slot.items()},
               **{f"v6_rest_{k}": R[0][k] for k in ("align", "k", "G", "scale", "shrink")})
    out.mkdir(parents=True)
    shutil.copyfile(B.OUT, out / "submission.py")
    for p in src.iterdir():
        if p.suffix == ".pt":
            shutil.copyfile(p, out / p.name)
    for f in eye + rest:
        shutil.copyfile(mem / f, out / f)
    np.savez(out / "weights.npz", **new)
    back = dict(np.load(out / "weights.npz", allow_pickle=False))
    assert all(back[k].dtype == v.dtype and np.array_equal(back[k], v) for k, v in z.items()), "a v5.1b array changed"
    load_model(out, None)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    Path(f"{out}.sha256").write_text("".join(f"{sha(p)}  {p.name}\n" for p in sorted(out.iterdir())))
    print(f"built {out}: {len(list(out.iterdir()))} files, {sum(p.stat().st_size for p in out.iterdir()) / 1e6:.1f} MB; "
          f"eye {eye}, rest {rest}, ft members {new['v6_ft_member'].tolist()}, gC5 t0 {new['v6_ft_t0']} every {new['v6_ft_every']} steps {new['v6_ft_steps']} lr {float(new['v6_ft_lr'])} on {int(new['v6_ft_on'])} labels {'v6r' if labels else 'v51b'} routed {routed} bundle {B.OUT.name}/{B.OUT.parent.name}")


if __name__ == "__main__":
    main()
