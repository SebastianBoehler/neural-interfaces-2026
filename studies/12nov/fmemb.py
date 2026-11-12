"""FMEMB (12 Nov): FROZEN foundation-model embeddings of every public window. No training; only a linear sanity probe.
One model per call -> runs/fmemb/<model>.npz, rows in data/proteus_windows.npz order (= ownspace.py O.PER / O.SES / O.y).
Embedding = mean over the non-CLS tokens of the last (`emb*`) / middle (`emb_mid*`) encoder block, float32 (2655, d).
Input views (the model-specific part is the fine-tuning code's: deep.build / steeg.py):
  emb, emb_mid          "cea": causal EA exactly as deep.AlignedNet fed a held-out person in all_loso: window t whitened
                        by the mean covariance of windows 0..t of its session + K=5 pseudo-windows of the other nine
                        people's mean covariance. Causal within the session, no label, nothing of the person's later data.
  emb_ea, emb_mid_ea    whole-session EA (the fine-tuning's training-row input). NOT causal.
  emb_raw, emb_mid_raw  each window alone: X / global RMS (cbramod, labram: deep's align=False) or per-window
                        per-channel z-scores (reve, steeg: the models' native input, as scripts/extract_reve.py).
  emb_rand              view cea through the same architecture at random init (seed 0): the no-op control. The
                        random-init control is probed for every view and layer (meta probe emb_rand*); only this one is saved.
Then per model: cbramod, labram: linear resample 480 -> 800 (200 Hz), x 0.15; reve: 480 -> 800 linear, clamp +-15,
positions by channel name; steeg: FFT resample 480 -> 512 (128 Hz), clamp +-15, channel table weights/chan_idx.json.
Also saved: y, person, session, interface, meta (JSON: checkpoint, sha256, load report, probes with per-person values, job).
Usage (Galvani): sbatch -J fmemb-MODEL -o runs/fmemb/logs/%x-%j.out scripts/cloud/py_any.sbatch studies/12nov/fmemb.py MODEL
       MODEL in cbramod | labram | reve | steeg. Weights must already be on Galvani (HF cache, runs/diag/steeg/weights).
"""
import hashlib, json, os, sys
from pathlib import Path
WS = Path("/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026")
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
os.environ["HF_HUB_OFFLINE"] = "1"  # never download: a model whose weights are not cached fails here
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import resample  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
import deep  # noqa: E402
import protocol  # noqa: E402
from features import covariances, invsqrtm  # noqa: E402

OUT, ST, K, N = WS / "runs/fmemb", WS / "runs/diag/steeg", 5.0, 2655
HUB = {"cbramod": deep.PRETRAINED["cbramod"][:2], "labram": deep.PRETRAINED["labram"][:2],
       "reve": ("brain-bzh/reve-base", deep.REVE_REV)}
HEAD = ("final_layer", "head")  # the only parameters allowed to differ from the checkpoint


def views(d, arch):
    """-> {view: float32 (N, 41, 480)}; see the module docstring."""
    X, C, per = d["X"].astype(np.float64), covariances(d["X"]), d["subject"]
    out = {v: np.empty(X.shape, np.float32) for v in ("cea", "ea")}
    for s in dict.fromkeys(d["stream"]):
        r = np.flatnonzero(d["stream"] == s); assert (np.diff(r) == 1).all(), s
        G = C[per != per[r[0]]].mean(0); n = np.arange(1, len(r) + 1)[:, None, None]
        out["cea"][r] = np.einsum("nij,njt->nit", invsqrtm((K * G + np.cumsum(C[r], 0)) / (K + n)), X[r])
        out["ea"][r] = np.einsum("ij,njt->nit", invsqrtm(C[r].mean(0)), X[r])
    out["raw"] = ((X - X.mean(-1, keepdims=True)) / np.maximum(X.std(-1, keepdims=True), 1e-6) if arch in ("reve", "steeg")
                  else X / np.sqrt(np.trace(C.mean(0)) / X.shape[1])).astype(np.float32)
    return out


def build(arch, pretrained, names):
    """-> (embed(float32 array (b, 41, 480)) -> [last, mid] pooled arrays, inner net, number of blocks)."""
    torch.manual_seed(0)
    if arch == "steeg":
        keep = deep.build; sys.path.insert(0, str(ST / "ours")); import steeg; deep.build = keep  # steeg.py patches deep.build
        tab = json.loads((ST / "weights/chan_idx.json").read_text()); assert tab["names"] == names and not tab["missing"]
        wrap = net = steeg.STEEG(tab["idx"], dp=0.0); blocks = net.blocks
        if pretrained:
            steeg.load_pretrained(net)
        prep = lambda x: np.clip(resample(x, 512, axis=-1), -15, 15).astype(np.float32)  # noqa: E731  (= STNet._aligned)
    else:
        wrap = deep.build(arch, 41, 480, pretrained); net, prep = wrap.net, (lambda x: x)
        blocks = {"cbramod": lambda: net.encoder.layers, "labram": lambda: net.blocks, "reve": lambda: net.transformer.layers}[arch]()
    wrap.to("cuda").eval(); L, got = len(blocks), {}; assert not any(m.training for m in wrap.modules())
    if arch == "reve":  # blocks are (attention, feed-forward) pairs: take the residual stream from the model itself
        wrap.fwd["return_output"] = True
    else:
        for name, i in (("mid", L // 2 - 1), ("last", L - 1)):
            blocks[i].register_forward_hook(lambda m, a, o, name=name: got.__setitem__(name, o))

    @torch.no_grad()
    def embed(x):
        got.clear(); o = wrap(torch.from_numpy(prep(x)).to("cuda"))
        if arch == "reve":
            assert isinstance(o, list) and len(o) == L + 1, type(o); got.update(mid=o[L // 2], last=o[L])
        res = []
        for k in ("last", "mid"):
            t = got[k]; t = t.flatten(1, 2) if arch == "cbramod" else t[:, 1:] if arch in ("labram", "steeg") else t
            assert t.ndim == 3 and t.shape[1] % 41 == 0, (k, tuple(got[k].shape))  # (b, 41 x patches, d)
            embed.tokens = tuple(t.shape[1:]); res.append(t.mean(1).float().cpu().numpy())
        return res
    return embed, net, L


def verify(arch, net, pretrained):
    """Every encoder parameter must equal the checkpoint tensor (pretrained) or almost none may (random init).
    A longer checkpoint tensor (LaBraM's temporal embedding) is compared on its leading slice."""
    if arch == "steeg":
        f = ST / "weights/small_encoder.pt"; ck = torch.load(f, map_location="cpu", weights_only=True)
    else:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        f = Path(hf_hub_download(HUB[arch][0], "model.safetensors", revision=HUB[arch][1])); ck = load_file(f)
    own = {k: v.detach().cpu() for k, v in net.named_parameters() if not k.startswith(HEAD)}
    same = [k for k, v in own.items() if k in ck and ck[k].ndim == v.ndim
            and torch.equal(ck[k][tuple(slice(0, s) for s in v.shape)].to(v.dtype), v)]
    bad = [sorted(set(own) - set(ck)), sorted((set(own) & set(ck)) - set(same)),
           sorted(k for k in set(ck) - set(net.state_dict()) if not k.startswith(HEAD))]  # missing, differ, unexpected
    rep = dict(checkpoint=str(f.resolve()), sha256=hashlib.sha256(f.read_bytes()).hexdigest(), checkpoint_tensors=len(ck),
               encoder_params=len(own), equal=len(same), n_values=int(sum(v.numel() for v in own.values())),
               missing=len(bad[0]), differ=len(bad[1]), unexpected=len(bad[2]), unexpected_names=bad[2][:12])
    print("LOAD", "pretrained" if pretrained else "random", json.dumps(rep), flush=True)
    assert len(same) == len(own) if pretrained else len(same) < 0.2 * len(own), "state dict does not match"
    return rep


def probe(E, d):
    """Leave-one-person-out balanced accuracy (%) of each of the ten people; embeddings z-scored within each session."""
    Z, per, acc = np.empty(E.shape), d["subject"], []
    for s in dict.fromkeys(d["stream"]):
        r = d["stream"] == s; Z[r] = (E[r] - E[r].mean(0)) / (E[r].std(0) + 1e-9)
    for p in sorted(set(per)):
        clf = LogisticRegression(C=0.1, max_iter=3000).fit(Z[per != p], d["y"][per != p])
        acc.append(balanced_accuracy_score(d["y"][per == p], clf.predict(Z[per == p])))
    return [round(100 * float(a), 2) for a in acc]


def main():
    arch = sys.argv[1]; assert arch in ("cbramod", "labram", "reve", "steeg") and torch.cuda.is_available(), arch
    d = protocol.load(); names = [str(c) for c in d["ch_names"]]
    assert len(d["y"]) == N and (d["split"] != "external").all() and d["X"].shape[1:] == (41, 480)
    V, res = views(d, arch), {}
    meta = dict(model=arch, job=os.environ.get("SLURM_JOB_ID"), gpu=torch.cuda.get_device_name(0), K=K,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), views={k: float(v.std()) for k, v in V.items()})
    for pre in (True, False):
        embed, net, L = build(arch, pre, names); meta["load" if pre else "load_rand"] = verify(arch, net, pre)
        a, b = embed(V["cea"][:8])[0], embed(V["cea"][:8])[0]; assert np.allclose(a, b, rtol=1e-4, atol=1e-6), "forward pass is not deterministic"
        for v in ("cea", "ea", "raw"):
            E = [embed(V[v][i:i + 32]) for i in range(0, N, 32)]
            for j, layer in enumerate(("", "_mid")):
                key = ("emb" if pre else "emb_rand") + layer + ("" if v == "cea" else "_" + v)
                res[key] = np.concatenate([e[j] for e in E]).astype(np.float32)
        meta.update(blocks=L, last_block=L, mid_block=L // 2, tokens=embed.tokens)
        del embed, net; torch.cuda.empty_cache()
    meta["dim"] = res["emb"].shape[1]; meta["probe"], meta["probe_person"], meta["stats"] = {}, {}, {}
    for k, E in res.items():
        assert E.shape == (N, meta["dim"]) and np.isfinite(E).all(), k
        assert len(np.unique(E, axis=0)) == N and (E.std(0) > 0).mean() > 0.9, f"{k}: embeddings do not differ between windows"
        meta["stats"][k] = dict(mean_abs=float(np.abs(E).mean()), std_over_windows=float(E.std(0).mean()))
        meta["probe_person"][k] = probe(E, d); meta["probe"][k] = round(float(np.mean(meta["probe_person"][k])), 2)
        print("PROBE", arch, k, meta["probe"][k], flush=True)
    gap = float(np.abs(res["emb"] - res["emb_rand"]).mean() / np.abs(res["emb"]).mean())
    meta["rel_diff_pretrained_vs_random"] = gap; assert gap > 0.05, f"pretrained and random-init embeddings coincide ({gap})"
    OUT.mkdir(parents=True, exist_ok=True); res = {k: v for k, v in res.items() if "rand" not in k or k == "emb_rand"}
    np.savez(OUT / f"{arch}.npz", **res, y=d["y"], person=np.array([s[-3:] for s in d["subject"]]),
             session=d["session"].astype(str), interface=d["interface"].astype(str), meta=json.dumps(meta))
    print("META", json.dumps(meta), flush=True); print("SAVED", OUT / f"{arch}.npz", {k: v.shape for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
