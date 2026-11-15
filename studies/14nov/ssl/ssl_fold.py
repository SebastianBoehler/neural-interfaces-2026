"""s277 one fold (= one held-out PROTEUS person, all_loso index i): SSL pretraining arms 1 (PROTEUS nine people) and 2 (+externals), arm 3
(supervised EEGNet on the nine people, labelled), arm 0 (untrained, same code path), then the frozen-trunk LR probe on the held-out person.
Usage: ssl_fold.py FOLD [smoke]  -> runs/probe/fold{FOLD}.json, checkpoints runs/ckpt/f{FOLD}_arm{a}.pt (not committed)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script dir
from common import *
import torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from benchmark_utils.metrics import balanced_accuracy

FOLD = int(sys.argv[1]); SMOKE = len(sys.argv) > 2 and sys.argv[2] == "smoke"
STEPS = 30 if SMOKE else 6000; BS = 256; SEED = 0; T0 = time.time()
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def aug(x):
    n, C, T = x.shape; g = x.device
    s = torch.randint(-60, 61, (n,), device=g); ix = (torch.arange(T, device=g)[None] + s[:, None]) % T
    x = torch.gather(x, 2, ix[:, None, :].expand(n, C, T))
    x = x * (0.7 + 0.7 * torch.rand(n, 1, 1, device=g)) + 0.2 * torch.rand(n, 1, 1, device=g) * torch.randn_like(x)
    return x * (torch.rand(n, C, 1, device=g) > 0.1)


def pretrain(X, sub_ids, arm, nC):
    net = build_net(nC, SEED + arm).to(DEV); tr = Trunk(net).to(DEV); D = trunk_dim(tr.cpu(), nC); tr.to(DEV).train()
    proj = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 64)).to(DEV)
    dec = nn.Sequential(nn.Linear(D, 512), nn.GELU(), nn.Linear(512, nC * NT)).to(DEV)
    ps = [*tr.parameters(), *proj.parameters(), *dec.parameters()]
    opt = torch.optim.AdamW(ps, 1e-3, weight_decay=1e-2); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=STEPS)
    g = torch.Generator(device="cpu").manual_seed(SEED + arm); N = len(X); stats = []
    for st in range(STEPS):
        b = torch.randint(0, N, (BS,), generator=g).to(DEV); x = X[b].float()
        m = torch.zeros(BS, 1, NT, device=DEV); pw = NT // 8
        for k in range(8):  # mask ~25 % of 8 time patches, all channels
            sel = torch.rand(BS, device=DEV) < 0.25; m[sel, :, k * pw:(k + 1) * pw] = 1
        h = tr(torch.cat([aug(x), aug(x), x * (1 - m)]))
        z = F.normalize(proj(h[:2 * BS]), dim=1); sim = z @ z.T / 0.2; sim.fill_diagonal_(-1e9)
        tgt = torch.cat([torch.arange(BS, 2 * BS), torch.arange(0, BS)]).to(DEV)
        rec = dec(h[2 * BS:]).view(BS, nC, NT); mm = m.expand(-1, nC, -1)
        loss_r = (((rec - x) ** 2) * mm).sum() / mm.sum()
        loss = F.cross_entropy(sim, tgt) + loss_r
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if st % max(STEPS // 10, 1) == 0: log(f"arm{arm} step {st} nce {loss.item() - loss_r.item():.3f} rec {loss_r.item():.3f}")
    return tr.eval()


def supervised(X, y, nC):
    net = build_net(nC, SEED + 3).to(DEV); tr = Trunk(net).to(DEV); net.train()
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=STEPS)
    g = torch.Generator(device="cpu").manual_seed(SEED + 3); yt = torch.from_numpy(y).to(DEV)
    for st in range(STEPS):
        b = torch.randint(0, len(X), (BS,), generator=g).to(DEV)
        loss = F.cross_entropy(net(aug(X[b].float())), yt[b]); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    log("arm3 final train ce", round(loss.item(), 3)); return tr.eval()


@torch.no_grad()
def feats(tr, X):
    return torch.cat([tr(torch.from_numpy(X[i:i + 1024]).to(DEV)).cpu() for i in range(0, len(X), 1024)]).numpy()


def probe(Ftr, ytr, ptr, Fte, yte):
    sc = StandardScaler().fit(Ftr); A, B = sc.transform(Ftr), sc.transform(Fte)
    best = max((np.mean([balanced_accuracy(ytr[v], LogisticRegression(C=C, max_iter=300).fit(A[t], ytr[t]).predict(A[v]))
                         for t, v in GroupKFold(3).split(A, ytr, ptr)]), C) for C in (1e-3, 1e-2, 1e-1, 1.0))
    pred = LogisticRegression(C=best[1], max_iter=300).fit(A, ytr).predict(B)
    return balanced_accuracy(yte, pred), best[1]


d = protocol.load(); name, tri, tei, held = fold_info(d, FOLD)
meta = json.load(open(EXT / "meta.json")); chans = meta["chans"]; nC = len(chans)
assert [str(c) for c in d["ch_names"][chans]] == meta["names"], "channel list differs from the export"
if SMOKE: tri, tei = tri[::6], tei[::3]
Xp = d["X"][:, chans].astype(np.float32); y = d["y"].astype(np.int64); subj = d["subject"].astype(str); st = d["stream"]
log(f"fold {FOLD} held {held}: train rows {len(tri)} test rows {len(tei)}, {nC} channels")
# pretraining pools (no unlabelled window of the held-out person in any of them: person-id asserts)
Xe = np.load(EXT / "X.npy", mmap_mode="r"); pe = np.load(EXT / "pid.npy"); people = meta["people"]
assert held not in set(people) and not any(p.startswith("Dreyer2026Proteus") for p in people)
ne = 3000 if SMOKE else len(Xe); Xe = np.ascontiguousarray(Xe[:ne]); pe = pe[:ne]
Xtr_nc = ea_noncausal(Xp[tri], st[tri])
pool1 = torch.from_numpy(Xtr_nc.astype(np.float16)).to(DEV); sub1 = set(subj[tri])
pool2 = torch.cat([pool1, torch.from_numpy(Xe).to(DEV)]); sub2 = sub1 | {people[i] for i in np.unique(pe)}
assert held not in sub1 and held not in sub2 and len(sub1) == 9 and set(subj[tei]) == {held}
# probe inputs: causal EA with the training people's mean raw covariance as prior, as the nets
G = covs(torch.from_numpy(Xp[tri]).to(DEV)).mean(0)
Xall_tr = ea_causal(Xp[tri], st[tri], G); Xall_te = ea_causal(Xp[tei], st[tei], G)
trunks = {0: Trunk(build_net(nC, SEED)).to(DEV).eval(), 1: pretrain(pool1, sub1, 1, nC), 2: pretrain(pool2, sub2, 2, nC), 3: supervised(pool1, y[tri], nC)}
res = dict(fold=FOLD, held=held, n_train=int(len(tri)), n_test=int(len(tei)), nC=nC, steps=STEPS, n_ext=int(len(Xe)), n_pool1=int(len(pool1)), arms={})
Path(CK).mkdir(parents=True, exist_ok=True); Path(OUT).mkdir(parents=True, exist_ok=True)
for a, tr in trunks.items():
    ck = CK / f"f{FOLD}_arm{a}{'_smoke' if SMOKE else ''}.pt"; torch.save(tr.net.state_dict(), ck)
    Ftr, Fte = feats(tr, Xall_tr), feats(tr, Xall_te)
    ba, C = probe(Ftr, y[tri], subj[tri], Fte, y[tei])
    res["arms"][a] = dict(ba=ba, C=C, ckpt_md5=md5(ck), feat_std=float(Ftr.std()))
    log(f"arm{a} BA {ba:.4f} C {C} md5 {res['arms'][a]['ckpt_md5'][:8]}")
assert len({res["arms"][a]["ckpt_md5"] for a in res["arms"]}) == 4, "two arms share a checkpoint"
res["pool_subjects_md5"] = {k: hashlib.md5("|".join(sorted(v)).encode()).hexdigest() for k, v in (("arm1", sub1), ("arm2", sub2))}
res["classes_in_test"] = np.bincount(y[tei], minlength=3).tolist()
json.dump(res, open(OUT / f"fold{FOLD}{'_smoke' if SMOKE else ''}.json", "w"), indent=1)
