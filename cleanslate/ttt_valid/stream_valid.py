"""Independent TTT validation streamer for team C packages (code of cleanslate/teamC is imported unchanged).

python stream_valid.py --pkg DIR --mode scherer|ret --held A|P04 --orders 0,..,7 --variants A0,A1,A2,A2h --out F.npz

scherer: held = Scherer2015 person letter, both visits, full-session shuffle of niv/orders.py (seed 71000+k, rng [seed,0,0]
         over all 18 visits in data order = the NI-V recipe); package trained on the ten PROTEUS people only.
ret:     held = P04|P10, sessions 2,3 only (package must not have trained on them), harness data.delivery over all
         PROTEUS rows (seed 71000+k), full-session shuffle.
Every visit/session = copy.deepcopy(model) + reset_state(), one window per predict(). Saves per variant/order the label,
posterior, seconds and the delivered position; labels y are saved for scoring (this is a dev/validation instrument).
"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "niv"), str(ROOT / "cleanslate/harness"), str(ROOT / "cleanslate/teamC")]
import orders as NO  # noqa: E402
import data as HD  # noqa: E402
from stream_eval import VARIANTS, load_model  # noqa: E402  (teamC, unchanged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--mode", required=True, choices=["scherer", "ret"])
    ap.add_argument("--held", required=True)
    ap.add_argument("--orders", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--variants", default="A0,A1,A2,A2h")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scherer", default=str(ROOT / "data/scherer2015/windows.npz"))
    ap.add_argument("--proteus", default=str(ROOT / "data/proteus_windows.npz"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda" or a.mode == "x", "GPU required (timing and speed)"
    tj = json.load(open(Path(a.pkg) / "train.json"))
    if a.mode == "scherer":
        z = np.load(a.scherer)
        X, y, subj, sess = z["X"], z["y"], z["subject"], z["session"]
        person = np.array([s.rsplit("/", 1)[-1] for s in subj])
        stream = np.char.add(np.char.add(subj, "|"), sess)
        iface = z["interface"]
        assert set(tj["people"]) == {f"P{i:02d}" for i in range(1, 11)} and "streams" in tj
        assert not any(s.startswith("Scherer") or s.split("|")[0] not in tj["people"] for s in tj["streams"]), "Scherer in training"
        visits = [s for s in dict.fromkeys(stream) if person[stream == s][0] == a.held]
        assert len(visits) == 2, visits
        plan = lambda seed: NO.permutation(stream, np.zeros(len(y), int), "full", seed)  # noqa: E731
    else:
        d = HD.load(a.proteus)
        X, y, person, stream, iface = d["X"], d["y"], d["person"], d["stream"], d["interface"]
        visits = [f"{a.held}|{s}" for s in ("2", "3")]
        assert not set(visits) & set(tj["streams"]), "a streamed session was trained on"
        assert a.held + "|1" in tj["streams"], "returning person not enrolled from session 1"
        plan = lambda seed: HD.delivery(stream, seed)  # noqa: E731
    out, meta = {}, []
    for k in map(int, a.orders.split(",")):
        p = plan(71000 + k)
        rows = np.concatenate([p[stream[p] == s] for s in visits])
        out[f"rows|{k}"] = rows
        for v in a.variants.split(","):
            model = load_model(a.pkg, dev, VARIANTS[v])
            lab, post, sec = [], [], []
            for s in visits:
                m = copy.deepcopy(model)
                m.reset_state()
                for r in rows[stream[rows] == s]:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    yy = m.predict(torch.as_tensor(X[r:r + 1]))
                    torch.cuda.synchronize()
                    sec.append(time.perf_counter() - t0)
                    assert yy.shape == (1,)
                    lab.append(int(yy[0])), post.append(m.last.detach().cpu().numpy())
            lab = np.array(lab)
            out[f"{v}|{k}|label"], out[f"{v}|{k}|post"], out[f"{v}|{k}|sec"] = lab, np.array(post), np.array(sec)
            meta.append(dict(variant=v, order=k, acc=round(float((lab == y[rows]).mean()), 4), ms=round(1e3 * float(np.mean(sec)), 1)))
            print("PASS", a.mode, a.held, meta[-1], flush=True)
    out["y"], out["stream"], out["iface"], out["person"] = y, stream, iface, person
    np.savez(a.out, **out)
    json.dump(dict(held=a.held, mode=a.mode, pkg=str(a.pkg), runs=meta, gpu=torch.cuda.get_device_name()), open(a.out + ".json", "w"))


if __name__ == "__main__":
    main()
