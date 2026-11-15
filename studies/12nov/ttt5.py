"""TTT5 (12 Nov, ledger s101): refinements of TTT4's fix20 (pseudo-labels on augmented inputs + consistency).
Variants (package form): fixA = fix20 replicated; fixW / fixS weaker / stronger augmentation; fixPL pseudo-labels on
augmented inputs only (no consistency); fixT only windows whose chain confidence is >= 0.6; fix40 40 steps; fixLR lr 2.5e-4.
Saved to runs/diag/ttt/probs5. (TTT4 docstring follows.) TTT4 (12 Nov, ledger s91): TTT3 with unsupervised consistency / entropy terms that also act on the hard windows.
Variants (package form, eval-mode SessionBN): ple20 (reference), fix20 (pseudo-labels on augmented inputs + consistency
between clean and augmented predictions on ALL windows), ent20 (pseudo-labels + entropy minimisation on all windows),
cons20 (consistency only, no labels). Saved to runs/diag/ttt/probs4.
(TTT3 docstring follows.) TTT3 (12 Nov, ledger s60): TTT2 with the PACKAGE FORM of fine-tuning next to the dev form, on the same trained nets.
Package form ("eval"): the net stays in eval mode while it is fine-tuned, so every SessionBN layer uses the causal
expanding statistics of the session (the formula of the shipped StreamBN traces) instead of batch statistics over
all windows so far; scope "conv" trains only the parameters a TorchScript trace exposes (no SessionBN affine).

(TTT2 docstring follows.) Per-session fine-tuning for every SessionBN member family and more variants.

For every fold a net is trained as usual. Each test session is then streamed: every STEP windows from T0 on, a copy
of the trained net is fine-tuned for N full-batch steps on all earlier windows of the session and predicts the next
STEP windows (SessionBN statistics stay causal). Label source for the fine-tuning:
  pl  the full chain's own causal outputs for those windows (argmax, weighted by confidence)  -> packageable
  or  the true labels                                                                         -> ceiling only
Saved proba (n, 3) per variant, rows sorted by index: base (no fine-tuning), pl20, or20, or60.
Usage (Leonardo/Galvani, from the work folder): python studies/12nov/ttt5.py TASK (arch = ORDER[TASK % 4], seed = TASK // 4) | smoke
"""
import copy, os, sys, time
from pathlib import Path
WS = Path(__import__("os").environ.get("NI26_WS", "/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026"))  # Leonardo: NI26_WS
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
import protocol  # noqa: E402
import session_bn as SB  # noqa: E402

HERE = Path(__file__).resolve().parent; OUT = WS / "runs/diag/ttt/probs5"
CH = np.load(os.environ.get("CHAIN_NPZ") or HERE / "chain_final.npz"); T0, STEP = 30, 10  # CHAIN_NPZ: other pseudo-labels (ttt9.py)
import hashlib  # noqa: E402
_LP = os.environ.get("CHAIN_NPZ") or str(HERE / "chain_final.npz")  # the label file's name and hash go into every output (review 13 Nov)
LABEL_TAG = f"{os.path.basename(_LP)}:{hashlib.sha256(open(_LP, 'rb').read()).hexdigest()[:12]}"
print("pseudo-labels from", _LP, LABEL_TAG, flush=True)
# variant -> (label source, steps, learning rate, parameter scope)
VARS = {"fixA": ("fix", 20, 5e-4, "all", "eval"), "fixW": ("fix", 20, 5e-4, "all", "eval"), "fixS": ("fix", 20, 5e-4, "all", "eval"),
        "fixPL": ("fixpl", 20, 5e-4, "all", "eval"), "fixT": ("fixt", 20, 5e-4, "all", "eval"), "fix40": ("fix", 40, 5e-4, "all", "eval"),
        "fixLR": ("fix", 20, 2.5e-4, "all", "eval")}
AUG = {"fixW": (5, 0.1, 0.05), "fixS": (30, 0.3, 0.2)}  # (max shift in samples, noise in channel sd, channel dropout); default (15, 0.2, 0.1)
ARCHS = {"fb": ("fb_sbn_rot05", 2.0), "sh": (None, 2.0), "fbraw": ("fbcnet_raw_sbn", 20.0), "fbl": ("fblight_ea_rot05_sbn", 5.0), "sinc": ("sincshallow_ea_ptm7r_rot05_sbn", 20.0),
         # s271 (15 Nov): structurally different families, NO pretrained trunk (none exists for these archs; fblight is the same), rot05 + SessionBN (k 20 as sinc/fbraw)
         "conf": ("conformer_ea_rot05_sbn", 20.0), "eegn": ("eegnet_ea_rot05_sbn", 20.0),
         "atc": ("atcnet_ea_rot05_sbn", 20.0), "deep4": ("deep4_ea_rot05_sbn", 20.0),
         "sincs": ("sincshallow_ea_rot05_sbn", 20.0),
         # s274 (15 Nov): four more braindecode families (all BatchNorm only -> fully SessionBN) and the EEGNet one-factor sweep (default = eegn)
         "incep": ("inceptionmi_ea_rot05_sbn", 20.0), "tcnet": ("eegtcnet_ea_rot05_sbn", 20.0),
         "ifnet": ("ifnet_ea_rot05_sbn", 20.0), "tsc": ("tsception_ea_rot05_sbn", 20.0),
         # s279 (B1): Fourier amplitude mixing across training people (p 0.5), otherwise the fb / eegn recipe
         "fb_am": ("fb_sbn_rot05_am", 2.0), "eegn_am": ("eegnet_ea_rot05_am_sbn", 20.0),
         "fb_am25": ("fb_sbn_rot05_am25", 2.0), "fb_am100": ("fb_sbn_rot05_am100", 2.0),  # s279 B1 undecided branch: p 0.25 / 1.0
         # s285: domain-generalisation losses on fb (CORAL weight 1, DANN lambda 0 -> 0.3, SWAD last 50 % of epochs; session_bn.make parses the suffix)
         "fb_coral": ("fb_sbn_rot05_coral", 2.0), "fb_dann": ("fb_sbn_rot05_dann", 2.0), "fb_swad": ("fb_sbn_rot05_swad", 2.0),
         **{f"eegn_{t}": (f"eegnet{t}_ea_rot05_sbn", 20.0) for t in ("k30", "k60", "k90", "f16", "d4", "p25", "p50")}}
for _f in ARCHS:  # a typo in a family key must fail at import, not after an hour on a GPU
    assert ARCHS[_f][0] is None or "_sbn" in ARCHS[_f][0], _f


def check_cfg(arch, net):
    """s274 guard: an EEGNet sweep family's kwargs are really in the trained net (kernel, F1, D, dropout), not silently the default."""
    if not arch.startswith("eegn_"):
        return
    import run_cv
    kw = {} if arch == "eegn_am" else run_cv.EEGN_SWEEP[arch[5:]];  # eegn_am (s279) is the default config
    ct = net.conv_temporal; f1 = ct.out_channels; d = net.conv_spatial.weight.shape[0] // f1
    got = dict(kernel_length=ct.kernel_size[-1], F1=f1, D=d, drop_prob=net.drop_1.p); want = {**dict(kernel_length=64, F1=8, D=2, drop_prob=0.25), **kw}
    assert got == want, (arch, got, want); print("cfg ok", arch, got, flush=True)  # sincs: from-scratch control of sinc (s271 amendment)
def check_am(arch, m, d, te, proto="all_loso"):
    """s279 guard: an *_am family really mixed (am set, windows mixed, batch changed) and no held-out person was a donor; others must have am 0."""
    if "_am" not in arch:
        assert not getattr(m, "am", 0), arch; return
    st = m.am_stats; held = set(np.unique(d["subject"][te]).tolist()); assert m.am == {"fb_am25": 0.25, "fb_am100": 1.0}.get(arch, 0.5) and st["mixed"] > 0 and st["absdiff"] > 0, (arch, st)
    if proto == "all_loso":  # seen_session trains on the held-out person's earlier sessions by design
        assert not (held & m.fit_subjects), f"held-out person(s) {held & m.fit_subjects} among the training people"
    print(f"am ok {arch}: mixed {st['mixed']} windows in {st['batches']} batches, mean |dx| per batch {st['absdiff'] / st['batches']:.1f}, "
          f"{len(m.fit_subjects)} training people, held-out {sorted(held)}", flush=True)


DGARM = {"fb_coral": ("coral", 1.0), "fb_dann": ("dann", 0.3), "fb_swad": ("swad", 1.0)}


def check_dg(arch, m, d, te, proto="all_loso"):
    """s285 guard: a DG arm applied exactly its own loss (setting equals the arm's, the loss was active and changed the batch's loss,
    person labels = the training people only, SWAD snapshots differ); every other family has all three at 0."""
    for k in ("coral", "dann", "swad"):
        assert getattr(m, k) == (DGARM[arch][1] if arch in DGARM and DGARM[arch][0] == k else 0.0), (arch, k, getattr(m, k))
    if arch not in DGARM:
        return
    st, k = m.dg_stats, DGARM[arch][0]; held = set(np.unique(d["subject"][te]).tolist()); assert st["steps"] > 0, (arch, st)
    if proto == "all_loso":
        assert not (held & m.fit_subjects), f"held-out person(s) {held & m.fit_subjects} among the training people"
    if k == "coral":
        assert st["coral_pairs"] > 0 and st["coral_pen"] > 0, (arch, st)
    if k == "dann":
        assert st["adv_on"] > 0 and st["adv_loss"] > 0, (arch, st)
    if k == "swad":
        assert st["snaps"] >= 2 and st["snap_diff"] > 0 and st["avg_diff"] > 0, (arch, st)
    print(f"dg ok {arch}: {st}, {len(m.fit_subjects)} training people, held-out {sorted(held)}", flush=True)


ORDER = os.environ["FAMS"].split(",") if os.environ.get("FAMS") else ["fbraw", "fbl", "sinc", "fb"]  # FAMS (s271): a family list, task = family + len * seed


def strong(X, g, shift=15, noise=0.2, drop=0.1):
    """time shift up to `shift` samples, Gaussian noise at `noise` of the channel sd, a share `drop` of channels zeroed"""
    sh = int(torch.randint(-shift, shift + 1, (1,), generator=g)); Xs = torch.roll(X, sh, -1)
    Xs = Xs + noise * Xs.std(-1, keepdim=True) * torch.randn(Xs.shape, generator=g).to(X.device)
    keep = (torch.rand(Xs.shape[0], Xs.shape[1], 1, generator=g) > drop).float().to(X.device); return Xs * keep


def make(arch, d, seed):
    cfg = ARCHS[arch][0]
    if cfg is not None:
        m = SB.make(cfg, d, seed)
        if seed >= 6 and "{seed}" in (getattr(m, "init", None) or ""):  # pretrained trunks exist for seeds 0-5 only (13 Nov: fresh sinc tasks died on a missing file)
            m.init = m.init.replace("{seed}", str(seed % 6)); print(f"{arch} seed {seed}: pretrained trunk of seed {seed % 6}, training seed {seed}", flush=True)
        return m
    m = SB.SessionNet(d, arch=SB.ARCH["sh"], seed=seed, **SB.CFGS["sbn"]); m.rot = 0.0; return m


def chain_labels(proto):
    if proto == "all_loso":
        return {i: CH["all_loso"][i] for i in range(len(CH["all_loso"]))}
    return {int(g): CH["seen_session"][k] for k, g in enumerate(CH["idx_seen"])}


def stream(m, X, y, Pc, seed):
    """-> {variant: (n, 3)} for one session; X aligned causal inputs (n, C, T) on the device."""
    net = m.net.eval(); n = len(X)
    with torch.no_grad():
        base = torch.softmax(net(X)[:, :3].double(), 1).cpu().numpy()
    out = {"base": base, **{v: base.copy() for v in VARS}}
    lab_pl = torch.from_numpy(Pc.argmax(1)).to(X.device); w_pl = torch.from_numpy(Pc.max(1)).float().to(X.device); lab_or = torch.from_numpy(y).to(X.device)
    soft = torch.from_numpy(Pc).float().to(X.device)
    for v, (src, steps, lr, scope, mode) in VARS.items():
        torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
        for t in range(T0, n, STEP):
            net2 = copy.deepcopy(net)
            sbn = {id(p) for mod in net2.modules() if isinstance(mod, SB.SessionBN) for p in mod.parameters()}
            if mode == "train":  # dev form: batch statistics over the windows so far
                for mod in net2.modules():
                    if isinstance(mod, SB.SessionBN):
                        mod.m, mod.momentum = t, 0.0
                net2.train()
            else:  # package form: causal expanding statistics, no dropout, no running-statistic updates
                net2.eval()
            if scope == "head":  # only the output layer and the SessionBN affine parameters
                last = [p for p in net2.parameters() if p.dim() > 1 and p.shape[0] >= 3][-1]
                params = [p for p in net2.parameters() if id(p) in sbn or p is last]
            elif scope == "conv":  # everything a TorchScript trace exposes as a parameter
                params = [p for p in net2.parameters() if id(p) not in sbn]
            else:
                params = list(net2.parameters())
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
            for _ in range(steps):
                opt.zero_grad(); z = net2(X[:t])[:, :3]
                if src == "soft":
                    loss = -(soft[:t] * F.log_softmax(z, 1)).sum(1).mean()
                elif src == "pl":
                    loss = (w_pl[:t] * F.cross_entropy(z, lab_pl[:t], reduction="none")).sum() / w_pl[:t].sum()
                elif src in ("fix", "fixpl", "fixt"):
                    zs = net2(strong(X[:t], gen, *AUG.get(v, (15, 0.2, 0.1))))[:, :3]
                    w = w_pl[:t] * (w_pl[:t] >= 0.6).float() if src == "fixt" else w_pl[:t]
                    loss = (w * F.cross_entropy(zs, lab_pl[:t], reduction="none")).sum() / w.sum().clamp_min(1e-6)
                    if src != "fixpl":
                        loss = loss + F.kl_div(F.log_softmax(zs, 1), F.softmax(z.detach(), 1), reduction="batchmean")
                else:
                    loss = F.cross_entropy(z, lab_or[:t])
                loss.backward(); opt.step()
            net2.eval(); hi = min(n, t + STEP)
            with torch.no_grad():
                out[v][t:hi] = torch.softmax(net2(X[:hi])[:, :3].double(), 1).cpu().numpy()[t:hi]
    return out


def run(arch, d, proto, seed, out, epochs=0):
    tag = f"ttt5_{arch}"; names = ["base"] + list(VARS)
    if all((out / f"{tag}_{v}-{proto}-s{seed}.npz").exists() for v in names):
        return print("exists", tag, proto, seed, flush=True)
    got = {v: [] for v in names}; rows_all = []; t0 = time.time(); Pc_all = chain_labels(proto)
    for fold, tr, te in protocol.folds(d, proto):
        m = make(arch, d, seed)
        if epochs:
            m.epochs = epochs
        m.fit(np.flatnonzero(tr)); m.set_k(ARCHS[arch][1]); rows = np.flatnonzero(te); st = d["stream"][rows]
        ref = protocol.predict_streams(m, d, te, proba=True)
        for s in dict.fromkeys(st):
            r = rows[st == s]; X = torch.from_numpy(m._aligned(r, causal=True)).to(m.dev)
            res = stream(m, X, d["y"][r], np.stack([Pc_all[int(g)] for g in r]), seed)
            for v in names:
                got[v].append(res[v])
            rows_all.append(r)
        base = np.concatenate(got["base"][-len(dict.fromkeys(st)):])
        print(f"{tag} s{seed} {proto} {fold} base-vs-harness max diff {np.abs(base - ref).max():.2e} {time.time() - t0:.0f}s", flush=True)
        if epochs:
            break
    idx = np.concatenate(rows_all); o = np.argsort(idx)
    for v in names:
        P = np.concatenate(got[v])[o]; assert np.isfinite(P).all()
        np.savez(out / f"{tag}_{v}-{proto}-s{seed}.npz", proba=P, index=idx[o])
        print(f"TOTAL {tag}_{v} s{seed} {proto} n={len(idx)} acc={np.mean(P.argmax(1) == d['y'][idx[o]]):.4f} {time.time() - t0:.0f}s", flush=True)


def main():
    d = protocol.load()
    if sys.argv[1] == "smoke":
        out = WS / "runs/diag/ttt/smoke5"; out.mkdir(parents=True, exist_ok=True)
        for arch in ("fbraw", "fbl", "sinc"):
            run(arch, d, "seen_session", 0, out, epochs=1)
        return print("SMOKE OK", flush=True)
    task = int(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
    for proto in ("all_loso", "seen_session"):
        run(ORDER[task % len(ORDER)], d, proto, task // len(ORDER), OUT)


if __name__ == "__main__":
    main()
