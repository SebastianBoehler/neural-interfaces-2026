"""s277 form 2 pieces: augmentation, pretraining (recon-only or recon+SimCLR), end-to-end fine-tuning, feature probe. Same code as ssl_fold.py plus weights and a step count argument."""
from common import *
import torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from benchmark_utils.metrics import balanced_accuracy
BS = 256


def aug(x):
    n, C, T = x.shape; g = x.device
    s = torch.randint(-60, 61, (n,), device=g); ix = (torch.arange(T, device=g)[None] + s[:, None]) % T
    x = torch.gather(x, 2, ix[:, None, :].expand(n, C, T))
    x = x * (0.7 + 0.7 * torch.rand(n, 1, 1, device=g)) + 0.2 * torch.rand(n, 1, 1, device=g) * torch.randn_like(x)
    return x * (torch.rand(n, C, 1, device=g) > 0.1)


def pretrain(X, steps, nce_w, seed, nC, log):
    """nce_w 0 = masked-patch reconstruction only (the contrastive branch is not computed), 1 = SimCLR + reconstruction."""
    net = build_net(nC, seed).to(DEV); tr = Trunk(net).to(DEV); D = trunk_dim(tr.cpu(), nC); tr.to(DEV).train()
    proj = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 64)).to(DEV)
    dec = nn.Sequential(nn.Linear(D, 512), nn.GELU(), nn.Linear(512, nC * NT)).to(DEV)
    opt = torch.optim.AdamW([*tr.parameters(), *proj.parameters(), *dec.parameters()], 1e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    g = torch.Generator(device="cpu").manual_seed(seed); pw = NT // 8
    for st in range(steps):
        b = torch.randint(0, len(X), (BS,), generator=g).to(DEV); x = X[b].float(); m = torch.zeros(BS, 1, NT, device=DEV)
        for k in range(8):
            sel = torch.rand(BS, device=DEV) < 0.25; m[sel, :, k * pw:(k + 1) * pw] = 1
        mm = m.expand(-1, nC, -1)
        if nce_w:
            h = tr(torch.cat([aug(x), aug(x), x * (1 - m)])); hm = h[2 * BS:]
            z = F.normalize(proj(h[:2 * BS]), dim=1); sim = z @ z.T / 0.2; sim.fill_diagonal_(-1e9)
            nce = F.cross_entropy(sim, torch.cat([torch.arange(BS, 2 * BS), torch.arange(0, BS)]).to(DEV))
        else:
            hm = tr(x * (1 - m)); nce = torch.zeros((), device=DEV)
        rec = (((dec(hm).view(BS, nC, NT) - x) ** 2) * mm).sum() / mm.sum()
        loss = nce_w * nce + rec; opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if st % (steps // 6) == 0: log(f"pre nce_w {nce_w} step {st} nce {nce.item():.3f} rec {rec.item():.3f}")
    return tr.eval()


def finetune(net, X, y, steps, seed, reinit_head=True):
    """The arm-3 recipe (CE, same augmentations, AdamW 1e-3 one-cycle) on a net, optionally from pretrained trunk weights with a fresh head."""
    torch.manual_seed(seed)
    if reinit_head:
        for mod in net.final_layer.modules():
            if hasattr(mod, "reset_parameters"): mod.reset_parameters()
    net.to(DEV).train(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps); g = torch.Generator(device="cpu").manual_seed(seed)
    yt = torch.from_numpy(y).to(DEV)
    for st in range(steps):
        b = torch.randint(0, len(X), (BS,), generator=g).to(DEV)
        loss = F.cross_entropy(net(aug(X[b].float())), yt[b]); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net.eval(), float(loss)


@torch.no_grad()
def predict(net, X):
    net.eval(); return torch.cat([net(torch.from_numpy(X[i:i + 1024]).to(DEV)).argmax(1).cpu() for i in range(0, len(X), 1024)]).numpy()


@torch.no_grad()
def feats(tr, X):
    return torch.cat([tr(torch.from_numpy(X[i:i + 1024]).to(DEV)).cpu() for i in range(0, len(X), 1024)]).numpy()


def probe(Ftr, ytr, ptr, Fte, yte):
    sc = StandardScaler().fit(Ftr); A, B = sc.transform(Ftr), sc.transform(Fte)
    best = max((np.mean([balanced_accuracy(ytr[v], LogisticRegression(C=C, max_iter=300).fit(A[t], ytr[t]).predict(A[v]))
                         for t, v in GroupKFold(3).split(A, ytr, ptr)]), C) for C in (1e-3, 1e-2, 1e-1, 1.0))
    return balanced_accuracy(yte, LogisticRegression(C=best[1], max_iter=300).fit(A, ytr).predict(B)), best[1]
