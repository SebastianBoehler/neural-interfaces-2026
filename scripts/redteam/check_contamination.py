import numpy as np, sys, re, collections
S = sys.argv[1]
ship = dict(np.load(f'{S}/pkg_cap250/weights.npz', allow_pickle=False))
b321 = dict(np.load(f'{S}/b321/P04/weights.npz', allow_pickle=False))
real = dict(np.load(f'{S}/prod/ret_P04/weights.npz', allow_pickle=False))
w = ship['member_weights'].astype(float); w = w / w.sum()
def group(k):
    m = re.match(r'net(\d+)_', k)
    if m:
        i = int(m.group(1)); return f'net (weight>0)' if w[1 + i] > 0 else 'net (zero weight)'
    return k.split('_')[0] if not k.startswith('v6_') else '_'.join(k.split('_')[:2])
def same(a, b):
    return a.shape == b.shape and (np.array_equal(a, b, equal_nan=True) if a.dtype.kind not in 'US' else (a == b).all())
tab = collections.defaultdict(lambda: [0, 0, 0, 0])
keep = collections.defaultdict(list)
for k in ship:
    g = group(k)
    e1 = k in b321 and same(ship[k], b321[k]); e2 = k in real and same(ship[k], real[k])
    tab[g][0] += 1; tab[g][1] += int(e1); tab[g][2] += int(e2)
    if e1 and not e2: keep[g].append(k)
print('%-18s %5s %14s %14s' % ('array group', 'n', '=ship in 321', '=ship in real'))
for g, v in sorted(tab.items()):
    print('%-18s %5d %14d %14d' % (g, v[0], v[1], v[2]))
print()
for g, ks in sorted(keep.items()):
    print(g, 'retained from the all-data package in the 321 bundle but refit in the real fold:', [k for k in ks][:40])
print('weights: core1 %.4f, hilbert core2 %.4f, sum nets %.4f, rest share v6_rest_w %.4f' % (w[0], w[-1], w[1:-1].sum(), float(ship['v6_rest_w'])))
print('active net slots', [i for i in range(len(w) - 2) if w[1 + i] > 0])
