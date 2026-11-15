"""s277 form 2, one fold (held-out PROTEUS person, all_loso index): 30000-step pretraining on shifted PROTEUS crops of the nine people + externals,
arms A (recon only) and B (SimCLR + recon); end-to-end fine-tuning (arm-3 recipe, 6000 steps, 3 seeds) of A, B and a random-init net (0f), plus the linear
probe of A and B. Usage: ssl_form2.py FOLD [smoke] -> runs/probe2/fold{FOLD}.json. The supervised-only reference (arm 3) is form 1's."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script dir
from ssl_nets import *

FOLD = int(sys.argv[1]); SMOKE = len(sys.argv) > 2 and sys.argv[2] == "smoke"
PRE = 40 if SMOKE else 30000; FT = 40 if SMOKE else 6000; SEEDS = (0,) if SMOKE else (0, 1, 2); T0 = time.time()
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)
OFFS = ["start0.5", "start2", "start3", "start4"]  # runs/crops/crop_<off>.npz: the 26 intersect channels, float16, rows in PROTEUS order (made by studies/14nov/ssl/mk_crops.py)
d = protocol.load(); name, tri, tei, held = fold_info(d, FOLD)
meta = json.load(open(EXT / "meta.json")); chans = meta["chans"]; nC = len(chans)
assert [str(c) for c in d["ch_names"][chans]] == meta["names"]
y = d["y"].astype(np.int64); subj = d["subject"].astype(str); st = d["stream"]
if SMOKE: tri, tei = tri[::6], tei[::3]
Xp = d["X"][:, chans].astype(np.float32)
# crops pool of the nine training people (every file has the PROTEUS row layout: subject/session/ch_names asserted equal), whole-session EA
crops, csub = [], set(subj[tri])
crops.append(ea_noncausal(Xp[tri], st[tri]).astype(np.float16))
for nm in OFFS:
    with np.load(WS / "runs/crops" / f"crop_{nm}.npz") as w:
        assert [str(c) for c in w["names"]] == meta["names"]; assert len(w["X"]) == len(subj), nm
        crops.append(ea_noncausal(w["X"][tri].astype(np.float32), st[tri]).astype(np.float16))  # same rows as the fold's training people
Xe = np.load(EXT / "X.npy", mmap_mode="r"); pe = np.load(EXT / "pid.npy"); people = meta["people"]
ne = 3000 if SMOKE else len(Xe); Xe = np.ascontiguousarray(Xe[:ne]); pe = pe[:ne]
assert held not in csub and held not in {people[i] for i in np.unique(pe)} and len(csub) == 9 and set(subj[tei]) == {held}
pool = torch.cat([torch.from_numpy(np.concatenate(crops)).to(DEV), torch.from_numpy(Xe).to(DEV)])
sup = torch.from_numpy(crops[0]).to(DEV); ysup = y[tri]  # labelled FT rows = the nine people's original windows
G = covs(torch.from_numpy(Xp[tri]).to(DEV)).mean(0)
Xtr_c = ea_causal(Xp[tri], st[tri], G); Xte_c = ea_causal(Xp[tei], st[tei], G)
log(f"fold {FOLD} held {held}: pool {len(pool)} ({len(Xe)} ext), labelled {len(sup)}, test {len(tei)}")
res = dict(fold=FOLD, held=held, pre_steps=PRE, ft_steps=FT, n_pool=int(len(pool)), n_ext=int(len(Xe)), n_labelled=int(len(sup)), probe={}, ft={})
Path(CK).mkdir(parents=True, exist_ok=True); Path(OUT.parent / "probe2").mkdir(parents=True, exist_ok=True)
pre = {"A": pretrain(pool, PRE, 0.0, 11, nC, log), "B": pretrain(pool, PRE, 1.0, 12, nC, log)}
for k, tr in pre.items():
    ck = CK / f"f2_{FOLD}_{k}{'_smoke' if SMOKE else ''}.pt"; torch.save(tr.net.state_dict(), ck)
    ba, C = probe(feats(tr, Xtr_c), ytr := y[tri], subj[tri], feats(tr, Xte_c), y[tei]); res["probe"][k] = dict(ba=ba, C=C, md5=md5(ck)); log(f"probe {k} {ba:.4f}")
sd = {k: {n: v.detach().cpu().clone() for n, v in tr.net.state_dict().items()} for k, tr in pre.items()}
for s in SEEDS:
    for arm in ("A", "B", "0f"):
        net = build_net(nC, 200 + s)  # same init seed for the random-init control and the pretrained arms' (re-initialised) head
        if arm != "0f": net.load_state_dict(sd[arm])
        w0 = {n: v.detach().cpu().clone() for n, v in net.state_dict().items() if v.dtype.is_floating_point}
        net, ce = finetune(net, sup, ysup, FT, s, reinit_head=arm != "0f")
        assert any((net.state_dict()[n].cpu() != v).any() for n, v in w0.items()), "fine-tuning changed nothing"
        ba = balanced_accuracy(y[tei], predict(net, Xte_c)); res["ft"][f"{arm}_s{s}"] = dict(ba=ba, train_ce=ce); log(f"FT {arm} seed {s} BA {ba:.4f} ce {ce:.3f}")
res["pretrained_differ"] = bool(any((sd["A"][n] != sd["B"][n]).any() for n in sd["A"] if sd["A"][n].dtype.is_floating_point))
assert res["pretrained_differ"]
json.dump(res, open(OUT.parent / "probe2" / f"fold{FOLD}{'_smoke' if SMOKE else ''}.json", "w"), indent=1)
