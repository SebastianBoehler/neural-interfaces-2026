import numpy as np, sys, warnings
from scipy.signal import butter, sosfiltfilt
warnings.filterwarnings('ignore')
S = sys.argv[1]
z = np.load('data/proteus_windows.npz', allow_pickle=True)
X, y = z['X'], z['y']; subj = np.array([s.split('/')[-1] for s in z['subject']]); ses = z['session'].astype(str); itf = z['interface'].astype(str); task = z['task'].astype(str)
w = dict(np.load(f'{S}/pkg_cap250/weights.npz', allow_pickle=False))
names = [str(n) for n in w['route_names']]
win = np.hanning(120)
def feat(x):
    x = np.asarray(x, np.float64); x = x - x.mean(0, keepdims=True); x = x - x.mean(1, keepdims=True)
    seg = np.stack([x[:, s:s + 120] for s in range(0, 361, 60)], 1) * win
    L = np.log((np.abs(np.fft.rfft(seg, axis=-1)) ** 2).mean(1)[:, 2:41] + 1e-12)
    return (L - L.mean(1, keepdims=True)).ravel()
F = np.stack([feat(x) for x in X])
# (b1) identity gallery rows rebuilt from the data, shipped package
sig = {(p, s): F[(subj == p) & (ses == s)].mean(0) for p, s in sorted(set(zip(subj, ses)))}
mu = np.mean(list(sig.values()), 0)
print('b1 bio_mu rebuilt from the 14 public sessions: max abs diff %.2e' % np.abs(mu - w['bio_mu']).max())
gal = np.stack([((np.mean([v for (p, s), v in sig.items() if p == n], 0) - mu) / w['bio_sd']) @ w['bio_V'] for n in names])
print('b1 bio_gal row i rebuilt for person i: max abs diff %.2e; nearest shipped row = own row for %d/10 people' % (
    np.abs(gal - w['bio_gal']).max(), sum(int(np.argmin(((w['bio_gal'] - g) ** 2).sum(1)) == i) for i, g in enumerate(gal))))
# (b2) own-space prior, band 0, 40 windows per person
lo, hi = w['own_bands'][0]; sos = butter(4, [float(lo), float(hi)], btype='band', fs=120., output='sos'); ok = 0
for i, n in enumerate(names):
    idx = np.flatnonzero(subj == n)[::max(1, (subj == n).sum() // 40)]
    C = 0
    for r in idx:
        x = X[r].astype(np.float64); b = sosfiltfilt(sos, x - x.mean(0, keepdims=True), axis=-1); c = b @ b.T / b.shape[-1]; C = C + c + 1e-3 * np.trace(c) * np.eye(41) / 41
    C /= len(idx)
    d = [np.linalg.norm(np.log(np.diag(C)) - np.log(np.diag(w['own_prior'][j, 0]))) for j in range(10)]
    ok += int(np.argmin(d) == i)
print('b2 own_prior (band %g-%g Hz): nearest shipped person prior = own person for %d/10 people' % (lo, hi, ok))
# (b3) personal head vs contrast gallery vs TSEM template prior, same 256-PC view
c = lambda a: (a - a.mean(0, keepdims=True)).ravel()
Wh, G, T = w['route_ps1_W'], w['route_gal'], w['v6_tpri']
S1 = np.array([[np.corrcoef(c(Wh[i]), c(G[j]))[0, 1] for j in range(10)] for i in range(10)])
S2 = np.array([[np.corrcoef(c(T[i]), c(G[j][:, :T.shape[2]]))[0, 1] for j in range(10)] for i in range(10)])
print('b3 head i vs gallery j: own person is the best match for %d/10 (diag mean %.2f, off-diag mean %.2f)' % ((S1.argmax(1) == np.arange(10)).sum(), np.diag(S1).mean(), (S1.sum() - np.trace(S1)) / 90))
print('b3 template prior i vs gallery j: own person best for %d/10 (diag mean %.2f, off-diag mean %.2f)' % ((S2.argmax(1) == np.arange(10)).sum(), np.diag(S2).mean(), (S2.sum() - np.trace(S2)) / 90))
# (c1) per-interface and per-run-type accuracy, real fold packages, held-new, stock
P = np.load(f'{S}/prod/stock/predictions.npz'); cells = np.load(f'{S}/prod/stock/cells.npz')
for mode in ('new', 'ret'):
    m = cells['mode'] == mode
    print('c1 %s cells BA by interface:' % mode, {i: round(float(cells['ba'][m & (cells['interface'] == i)].mean()), 2) for i in ('Graz', 'BrainHero')})
acc = {t: [] for t in sorted(set(task))}
for p in names:
    for o in range(71000, 71004):
        pr = P[f'new|{o}|{p}'].astype(int); m = subj == p
        for t in acc: acc[t].append(float((pr[m & (task == t)] == y[m & (task == t)]).mean()))
print('c1 held-new plain accuracy by run type:', {t: round(100 * float(np.mean(v)), 1) for t, v in acc.items()})
# (c2) can a model tell the interfaces apart? leave-one-person-out, label-free spectral features (64-d), raw and session-centred
from sklearn.linear_model import LogisticRegression
Z = ((F - w['bio_mu']) / w['bio_sd']) @ w['bio_V']
Zc = Z.copy()
for k in sig: 
    m = (subj == k[0]) & (ses == k[1]); Zc[m] -= Zc[m].mean(0)
for tag, A in (('raw', Z), ('session-centred', Zc)):
    for name, t in (('Graz vs BrainHero', (itf == 'Graz').astype(int)), ('acquisition vs online', np.char.startswith(task, 'Acq').astype(int))):
        a = []
        for p in names:
            tr = subj != p; clf = LogisticRegression(C=0.1, max_iter=500, class_weight='balanced').fit(A[tr], t[tr])
            pr = clf.predict(A[~tr]); a.append(np.mean([(pr[t[~tr] == k] == k).mean() for k in (0, 1)]))
        print('c2 %-22s %-16s LOPO balanced accuracy %.1f %% (min person %.1f)' % (name, tag, 100 * np.mean(a), 100 * np.min(a)))
