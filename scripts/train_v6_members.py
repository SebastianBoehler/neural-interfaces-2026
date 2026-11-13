"""Package weights for the v6 candidate (13 Nov plan; docs/V6_DESIGN.md): the eye net and the rest-class member, trained on
all public windows of the ten dev people (no held-out fold), several seeds.

TASK 0-2: eye net, studies/12nov/ocunet.py arm nomix (60 epochs), seed TASK; traces(d) = the 7 derived ocular traces with
          each session's running standardisation. Saves runs/v6_members/eye_s<seed>.pt (TorchScript, eval mode, input
          (B, 7, 480)) and checks the trace against the eager net on all public windows (< 1e-5).
TASK 3-5: rest net, studies/11nov/rest4/rest4.py arch sh, cond r1 (four outputs, rest loss weight 1; task windows plus the
          4 s before every Graz cue), seed TASK - 3, test prior k 2. Saves runs/v6_members/rest_s<seed>.pt (StreamBN
          TorchScript, as the package's other SessionBN members) and rest_s<seed>.npz (the InputStream keys align, k, G,
          scale, shrink). Checks: the package-form input stream equals the dev alignment, and the trace streamed over each
          public session equals the dev net's predict_proba_stream (softmax over the first three outputs; < 1e-4).
Usage (Leonardo .venv): python scripts/train_v6_members.py TASK | smoke"""
import importlib.util, json, os, sys, time
from pathlib import Path
ROOT = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402
from export_external_teacher import device_free_trace  # noqa: E402
OUT = ROOT / "runs/v6_members"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def eye(seed, smoke):
    O = load("ocunet", ROOT / "studies/12nov/ocunet.py"); d = protocol.load(); T = O.traces(d); allr = np.arange(len(d["y"]))
    torch.manual_seed(seed); t0 = time.time()
    # O.fit_predict trains and predicts; re-implemented minimally here only to keep the net (same loop, same seed use)
    g = torch.Generator().manual_seed(seed); net = O.Net().to(O.DEV); Xt, yt = torch.from_numpy(T), torch.from_numpy(d["y"])
    epochs = 1 if smoke else 60; nb = int(np.ceil(len(allr) / 64))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * nb)
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(allr), generator=g)
        for b in range(nb):
            j = perm[b * 64:(b + 1) * 64]
            if len(j) < 8:
                sched.step(); continue
            xb, yb = Xt[j].to(O.DEV), torch.nn.functional.one_hot(yt[j], 3).float().to(O.DEV)
            xb = xb * (0.8 + 0.4 * torch.rand(len(j), 1, 1, generator=g).to(O.DEV)) + 0.05 * torch.randn(xb.shape, generator=g).to(O.DEV)
            opt.zero_grad(); loss = -(yb * torch.log_softmax(net(xb), 1)).sum(1).mean(); loss.backward(); opt.step(); sched.step()
    net = net.cpu().eval(); x = torch.from_numpy(T[:4])
    with torch.no_grad():
        tr = torch.jit.trace(net, x, check_trace=False); X = torch.from_numpy(T)
        pe, pt = torch.softmax(net(X).double(), 1), torch.softmax(tr(X).double(), 1)
    diff = float((pe - pt).abs().max()); assert diff < 1e-5, f"eye trace differs from eager by {diff}"
    f = OUT / f"eye_s{seed}.pt"; tr.save(str(f))
    return dict(member="eye", seed=seed, file=f.name, epochs=epochs, trace_vs_eager=diff, train_acc=float((pe.argmax(1).numpy() == d["y"]).mean()), sec=round(time.time() - t0))


def rest(seed, smoke):
    R = load("rest4", ROOT / "studies/11nov/rest4/rest4.py"); d, d2, sel = R.data(); N = len(d["y"]); t0 = time.time()
    m = R.make("sh", "r1", d2, seed)
    if smoke:
        m.epochs = 1
    m.fit(np.arange(len(d2["y"]))); m.set_k(2.0); m.net.cpu().eval(); m.dev = "cpu"
    p = dict(align=np.bool_(m.align), k=np.float64(m.k), G=np.asarray(m.G, np.float64), scale=np.float64(m.scale), shrink=np.float64(0.001))
    assert m.align is True and not m.ref_clip and not m.ref_band, "the package InputStream has no clip / band reference"
    rows = np.flatnonzero(d["stream"] == d["stream"][0]); Xd = m._aligned(rows, True)
    # the package InputStream (submissions/ensemble_v5b: shrunk covariance, running mean blended with k pseudo-windows of G)
    cs, n, out = 0.0, 0, []
    for r in rows:
        x = d["X"][r].astype(np.float64); xc = x - x.mean(-1, keepdims=True); C = xc @ xc.T / x.shape[-1]
        C = 0.999 * C + 0.001 * np.trace(C) / len(C) * np.eye(len(C)); n += 1; cs = cs + C
        w, V = np.linalg.eigh((p["k"] * p["G"] + cs) / (p["k"] + n)); out.append(((V / np.sqrt(np.maximum(w, 1e-12))) @ V.T @ x).astype(np.float32))
    da = float(np.abs(np.stack(out) - Xd).max()); assert da < 1e-4 * max(1.0, float(np.abs(Xd).max())), f"input stream differs from dev alignment by {da}"
    tr = device_free_trace(SB.to_stream(m.net, 2.0), torch.from_numpy(np.ascontiguousarray(Xd[:4]))); SB.reset_stream(tr)
    ref = protocol.predict_streams(m, d, np.ones(N, bool), proba=True); worst = 0.0
    for s in dict.fromkeys(d["stream"]):
        r = np.flatnonzero(d["stream"] == s); SB.reset_stream(tr)
        with torch.no_grad():
            P = torch.softmax(tr(torch.from_numpy(m._aligned(r, True)))[:, :3].double(), 1).numpy()
        worst = max(worst, float(np.abs(P - ref[r]).max()))
    assert worst < 1e-4, f"rest trace differs from the dev net by {worst}"
    SB.reset_stream(tr); f = OUT / f"rest_s{seed}.pt"; tr.save(str(f)); np.savez(OUT / f"rest_s{seed}.npz", **p)
    return dict(member="rest", seed=seed, file=f.name, input_diff=da, trace_vs_dev=worst, in_sample_acc=float((ref.argmax(1) == d["y"]).mean()),
                rest_rows=len(sel), sec=round(time.time() - t0))


def main():
    smoke = sys.argv[1] == "smoke"; task = 0 if smoke else int(sys.argv[1])
    global OUT
    if smoke:
        OUT = ROOT / "runs/diag/v6_members_smoke"
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [0, 3] if smoke else [task]
    for t in tasks:
        res = eye(t, smoke) if t < 3 else rest(t - 3, smoke)
        print(json.dumps(res), flush=True)
    print("SMOKE OK" if smoke else "DONE", flush=True)


if __name__ == "__main__":
    main()
