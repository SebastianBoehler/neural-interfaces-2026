import numpy as np, glob, json, sys
from pathlib import Path
S = sys.argv[1]
z0 = np.load('data/proteus_windows.npz', allow_pickle=True)
y = z0['y']; itf = z0['interface']
ba = lambda yy, p: 100 * float(np.mean([(p[yy == c] == c).mean() for c in range(3) if (yy == c).any()]))
names = ['P%02d' % i for i in range(1, 11)]
rowsum = []
for d in sorted(glob.glob(S + '/laneA_out/*_P*_?')):
    n = Path(d).name
    z = np.load(f'{d}/{n}.npz'); j = json.load(open(f'{d}/{n}.json'))
    rows = z['rows']; yy = y[rows]
    idn, con = z['stock|identity'], z['stock|contrast']   # (T,3): j, score, ramp ; NaN when no decision?
    lab = z['stock|labels']; lab0 = z['zero|labels']
    T = len(rows)
    who = n.split('_')[1]; tj = names.index(who)
    def first(a):
        ok = np.flatnonzero(np.nan_to_num(a[:, 2]) > 0)
        return int(ok[0]) if len(ok) else -1
    fi, fc = first(idn), first(con)
    ri = np.nan_to_num(idn[:, 2]); rc = np.nan_to_num(con[:, 2])
    r = np.maximum(ri, rc)
    jj = np.where(ri >= rc, idn[:, 0], con[:, 0])
    routed = r > 0
    wrong = routed & (jj != tj)
    # identity score for the named person from window 10 on
    si = idn[10:, 1]; sc = con[:, 1]
    rowsum.append(dict(n=n, T=T, fi=fi, fc=fc, id_frac=float((ri > 0).mean()), con_frac=float((rc > 0).mean()), routed=float(routed.mean()), wrong=int(wrong.sum()),
                       id_named_true=float(np.nanmean(idn[10:, 0] == tj)), id_min=float(np.nanmin(si)), id_med=float(np.nanmedian(si)), id_end=float(si[-1]),
                       con_named_true=float(np.nanmean(con[~np.isnan(con[:, 0]), 0] == tj)) if (~np.isnan(con[:, 0])).any() else float('nan'),
                       con_med=float(np.nanmedian(sc)) if (~np.isnan(sc)).any() else float('nan'), con_end=float(sc[-1]),
                       ba=ba(yy, lab), ba_routed=ba(yy[routed], lab[routed]) if routed.sum() > 20 else float('nan'),
                       ba_unrouted=ba(yy[~routed], lab[~routed]) if (~routed).sum() > 20 else float('nan'), n_unrouted=int((~routed).sum()),
                       rec=j['stats']['stock']['recognition']))
print('%-14s %3s %4s %4s %5s %5s %5s %3s | %5s %6s %6s %6s | %5s %6s %6s | %5s %5s %5s %4s' % ('stream', 'T', 'fId', 'fCon', 'idFr', 'conFr', 'rout', 'wr', 'idTru', 'idMin', 'idMed', 'idEnd', 'cTru', 'cMed', 'cEnd', 'BA', 'BArt', 'BAun', 'nun'))
for r in rowsum:
    print('%-14s %3d %4d %4d %5.2f %5.2f %5.2f %3d | %5.2f %6.3f %6.3f %6.3f | %5.2f %6.3f %6.3f | %5.1f %5.1f %5.1f %4d' % (
        r['n'], r['T'], r['fi'], r['fc'], r['id_frac'], r['con_frac'], r['routed'], r['wrong'], r['id_named_true'], r['id_min'], r['id_med'], r['id_end'],
        r['con_named_true'], r['con_med'], r['con_end'], r['ba'], r['ba_routed'], r['ba_unrouted'], r['n_unrouted']))
import collections
for who in ('P04', 'P10'):
    for v in ('2', '3'):
        rs = [r for r in rowsum if r['n'].endswith(f'{who}_{v}')]
        print(who, v, 'BA %.2f' % np.mean([r['ba'] for r in rs]), 'identity ever fires %d/8' % sum(r['fi'] >= 0 for r in rs), 'first id', sorted(r['fi'] for r in rs),
              'first contrast', sorted(r['fc'] for r in rs), 'mean id-routed frac %.2f' % np.mean([r['id_frac'] for r in rs]), 'mean routed %.2f' % np.mean([r['routed'] for r in rs]))
