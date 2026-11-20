import numpy as np, sys, glob, json
from pathlib import Path
S = sys.argv[1]
d = np.load('data/proteus_windows.npz', allow_pickle=True)
y, itf = d['y'], d['interface']
def ba(yy, p): return 100 * float(np.mean([(p[yy == c] == c).mean() for c in range(3)]))
def cellba(rows, lab):
    return float(np.mean([ba(y[rows][itf[rows] == i], lab[itf[rows] == i]) for i in ('Graz', 'BrainHero')]))
print('REAL FOLD PACKAGES (production-20261120-32a94ab7, stock = cap250 recipe), full-shuffle orders 71000-71003')
for who in ('P04', 'P10'):
    r = np.load(f'{S}/prod/stock/pred_ret_{who}.npz'); n = np.load(f'{S}/prod/stock/pred_new_{who}.npz')
    for v in ('2', '3'):
        ret = [cellba(r[f'rows_{o}_{v}'], r[f'labels_{o}_{v}'].astype(int)) for o in range(71000, 71004)]
        new = [cellba(n[f'rows_{o}_{v}'], n[f'labels_{o}_{v}'].astype(int)) for o in range(71000, 71004)]
        print(f'  {who} visit {v}: enrolled-visit-1 package {np.mean(ret):6.2f}  {np.round(ret, 1)}   held-new package {np.mean(new):6.2f} {np.round(new, 1)}')
    new1 = [cellba(n[f'rows_{o}_1'], n[f'labels_{o}_1'].astype(int)) for o in range(71000, 71004)]
    print(f'  {who} visit 1 held-new {np.mean(new1):6.2f}')
print('LANE A CONDITIONAL REPLAY (stock arm), same visits')
for who in ('P04', 'P10'):
    for v in ('2', '3'):
        out = {}
        for dd in sorted(glob.glob(f'{S}/laneA_out/*_{who}_{v}')):
            nme = Path(dd).name; z = np.load(f'{dd}/{nme}.npz')
            out[nme.split('_')[0]] = cellba(z['rows'], z['stock|labels'])
        print(f'  {who} visit {v}: ' + ' '.join(f'{k} {v_:.1f}' for k, v_ in out.items()), ' mean full %.2f, within %.2f, runs %.2f' % (
            np.mean([v_ for k, v_ in out.items() if k.startswith('full')]), np.mean([v_ for k, v_ in out.items() if k.startswith('within')]), np.mean([v_ for k, v_ in out.items() if k.startswith('runs')])))
# confusion for real fold ret P04
for who in ('P04', 'P10'):
    r = np.load(f'{S}/prod/stock/pred_ret_{who}.npz')
    C = np.zeros((3, 3))
    for o in range(71000, 71004):
        for v in ('2', '3'):
            rows, lab = r[f'rows_{o}_{v}'], r[f'labels_{o}_{v}'].astype(int)
            for a, b in zip(y[rows], lab): C[a, b] += 1
    print(who, 'ret real-fold confusion rows=true MI/SUB/WORD (%):', np.round(100 * C / C.sum(1, keepdims=True), 1).tolist())
    # accuracy by delivery position
    acc = []
    for o in range(71000, 71004):
        for v in ('2', '3'):
            rows, lab = r[f'rows_{o}_{v}'], r[f'labels_{o}_{v}'].astype(int)
            acc.append((y[rows] == lab)[:180])
    acc = np.mean(acc, 0)
    print(who, 'ret real-fold plain accuracy by delivery position 0-9,10-19,20-39,40-59,60-99,100-139,140-179:', [round(100 * float(acc[a:b].mean()), 1) for a, b in ((0, 10), (10, 20), (20, 40), (40, 60), (60, 100), (100, 140), (140, 180))])
