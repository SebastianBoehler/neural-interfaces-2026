import numpy as np, collections
z = np.load('data/proteus_windows.npz', allow_pickle=True)
X, y = z['X'], z['y']
subj = np.array([s.split('/')[-1] for s in z['subject']]); ses = z['session']; itf = z['interface']; task = z['task']; run = z['run']; onset = z['onset']
print('names', list(z['ch_names']))
# 1 offset relation
for tag, sh in (('start0.5', -60), ('start2', 120), ('start3', 240)):
    w = np.load(f'data/proteus_windows_{tag}.npz', allow_pickle=True)
    Xa = w['X']
    assert (w['y'] == y).all(); print(tag, 'onset shift', np.unique(np.round(w['onset'] - onset, 3)))
    if sh < 0:   # other file starts earlier by -sh samples: default[:, :, :480+sh] == other[:, :, -sh:]
        d = np.abs(X[:, :, :480 + sh] - Xa[:, :, -sh:]).max()
    else:        # other file starts later: default[:, :, sh:] == other[:, :, :480-sh]
        d = np.abs(X[:, :, sh:] - Xa[:, :, :480 - sh]).max()
    print(tag, 'max abs diff under the stated shift', float(d))
    del w, Xa
# 2 scale
print('per-window channel std: median %.2f uV, 5%%/95%% %.2f/%.2f; abs max %.1f; mean of window means %.3f' % (
    np.median(X.std(-1)), *np.percentile(X.std(-1), [5, 95]), np.abs(X).max(), X.mean()))
# 3 PSD under fs=120
F = np.fft.rfftfreq(480, 1 / 120.)
P = (np.abs(np.fft.rfft(X * np.hanning(480), axis=-1)) ** 2).mean((0, 1))
for f in (2, 6, 10, 20, 30, 40, 45, 48, 49, 50, 51, 52, 55, 58, 59.75):
    i = np.argmin(np.abs(F - f)); print('  %5.2f Hz  %8.3f dB' % (F[i], 10 * np.log10(P[i])))
# 4 structure
print('sessions', len(set(zip(subj, ses))))
for (s, e) in sorted(set(zip(subj, ses))):
    m = (subj == s) & (ses == e)
    rows = []
    for r in sorted(set(run[m])):
        k = m & (run == r)
        rows.append(f"{r}:{task[k][0][:3]}{task[k][0][-2:]}:{k.sum()}:{np.bincount(y[k], minlength=3).tolist()}")
    print(s, e, m.sum(), ' '.join(rows))
print('onset diffs within run (s):', collections.Counter(np.round(np.diff(onset)[(np.diff(run) == 0) & (subj[1:] == subj[:-1])], 1)).most_common(8))
