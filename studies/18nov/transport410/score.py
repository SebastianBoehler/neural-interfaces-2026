"""Score sealed panels and split the original sign reversal into paired components."""
import json
from pathlib import Path

import numpy as np
from panel import BASE, DEV, MOTOR, OLD, ROOT, SOURCES, check, digest, old

ARMS = ('P','X','D','B')
GROUPS = [np.arange(4),np.arange(4,6),np.arange(6,8)]


def read(folder, fit, selected, people, orders):
    target = np.flatnonzero(np.isin(people, selected))
    card = json.loads((fit/'source.json').read_text())
    check(card['members'] == digest(fit/'members.npz') and card['source'] == digest(fit/'source.sha256')
          and not card['query_labels_opened'], 'members seal')
    with np.load(fit/'members.npz') as z:
        np.testing.assert_array_equal(target,z['index'])
        whole = {a:z[a].argmax(1) for a in ARMS}
    full = {a:[] for a in ('Z',)+ARMS}
    seals = []
    for i, order in enumerate(orders):
        path = folder/f'order_{i}.npz'
        record = json.loads(path.with_suffix('.json').read_text())
        check(record == dict(source=digest(folder/'source.sha256'),members=card['members'],
                             order=order,outputs=digest(path)), 'order seal')
        seals.append(record['outputs'])
        with np.load(path) as z:
            np.testing.assert_array_equal(target,z['index'])
            for a in full:
                full[a].append(z[a])
    return target,dict(whole={a:np.repeat(v[None,:],8,axis=0) for a,v in whole.items()},
                       full={a:np.stack(v) for a,v in full.items()}),seals


def mean_order(v):
    return np.mean([v[...,g].mean(-1) for g in GROUPS],axis=0)


def contrasts(v, orig):
    # Axes: six matched panels × four dev people plus its one source person × eight orders.
    src, dev = v[:,4,:],v[:,:4,:]
    rng = np.random.default_rng(410)
    n = 5000
    pi = rng.integers(6,size=(n,6))
    di = rng.integers(4,size=(n,4))
    sampled = {k:0. for k in ('source','five_dev','six_dev')}
    for g in GROUPS:
        # Crossed dev people use the same sampled order for their paired panel and original fit.
        od = rng.choice(g,size=(n,4,len(g)))
        os = rng.choice(g,size=(n,6,len(g)))
        sampled['source'] += src[pi[:,:,None],os].mean((1,2))/3
        sampled['five_dev'] += dev[pi[:,:,None,None],di[:,None,:,None],od[:,None,:,:]].mean((1,2,3))/3
        sampled['six_dev'] += orig[di[:,:,None],od].mean((1,2))/3
    val = dict(source=mean_order(src).mean(),five_dev=mean_order(dev).mean(),six_dev=mean_order(orig).mean())
    for a,left,right in [('matched','five_dev','source'),('training','six_dev','five_dev'),('original','six_dev','source')]:
        val[a] = val[left]-val[right]
        sampled[a] = sampled[left]-sampled[right]
    np.testing.assert_allclose(val['original'],val['matched']+val['training'],atol=1e-12)
    return {k:[float(v),*np.percentile(sampled[k],[10,90]).tolist()] for k,v in val.items()}


def main():
    check((ROOT/'replay.log').read_text().splitlines()[-1]=='OK','replay')
    assets,_,_ = old.source.setup()
    path = assets.CHAIN/'data/proteus_windows.npz'
    with np.load(path) as z:
        person = np.array([p.rsplit('/',1)[-1] for p in z['subject']])
        iface,session = z['interface'],z['session']
    with np.load(old.source.INPUT/'inputs.npz') as z:
        orders = z['orders'].tolist()
    panels = [read(ROOT/str(i),ROOT/str(i),DEV+(p,),person,orders) for i,p in enumerate(SOURCES)]
    original = read(MOTOR,OLD,DEV,person,orders)
    proofs = [s for _,_,s in panels+[original]]
    Path('before_labels.json').write_text(json.dumps(proofs))
    with np.load(path) as z:
        y = z['y']
    rowscores = []
    sourceold = json.loads((BASE/'ni26_burst407_loo_v2_20261118/loo.json').read_text())
    reportold = json.loads((MOTOR/'result.json').read_text())['results']
    for i,(rows,families,_) in enumerate(panels+[original]):
        data = {}
        held = DEV if i==6 else DEV+(SOURCES[i],)
        for family,byarm in families.items():
            for a,pred in byarm.items():
                metrics = {}
                for p in held:
                    sel = person[rows]==p
                    yy, ff, ss = y[rows][sel], iface[rows][sel], session[rows][sel]
                    hit = 100.*(pred[:,sel]==yy)
                    strat = {}
                    for f in sorted(set(ff)):
                        rr = ff==f
                        per_session = np.array([[hit[:,rr&(ss==s)&(yy==c)].mean(1) for c in range(3)]
                                                for s in sorted(set(ss[rr]))])
                        strat[f] = per_session.mean(0)
                    # Equal weight across official participant/session/interface cells.
                    cells = np.stack([strat[f] for f in sorted(strat)])
                    check(all(len(set(ss[ff==f]))==len(set(ss)) for f in strat), 'interface availability')
                    for f,val in strat.items():
                        metrics.setdefault(f,[]).append(val.mean(0))
                    for c in range(3):
                        metrics.setdefault(str(c),[]).append(cells[:,c,:].mean(0))
                    avg=cells.mean((0,1))
                    metrics.setdefault('all',[]).append(avg)
                    if family=='whole' and i<6 and p==SOURCES[i]:
                        np.testing.assert_allclose(avg,sourceold[p][a],atol=1e-10,rtol=0)
                    if i==6:
                        np.testing.assert_allclose(mean_order(avg),reportold[family]['metrics'][a]['people'][p],
                                                   atol=1e-10,rtol=0)
                for m,v in metrics.items():
                    data[family,a,m]=np.stack(v)
        rowscores.append(data)
    report={}
    for family in ('whole','full'):
        rows=rowscores[-1]
        controls=ARMS if family=='whole' else ('Z',)+ARMS
        for m in ('all','Graz','BrainHero','0','1','2'):
            for b in controls:
                if b=='X':
                    continue
                def delta(r):
                    return r[family,'X',m] - r[family,b,m]
                v=np.stack([delta(r) for r in rowscores[:-1]])
                o=delta(rows)
                stats=contrasts(v,o)
                if m=='all':
                    stats['matched_people'] = dict(zip(DEV+SOURCES,
                        np.r_[mean_order(v[:,:4]).mean(0),mean_order(v[:,4])].tolist()))
                report[f'{family}/X-{b}/{m}']=stats
    Path('result.json').write_text(json.dumps(dict(proof=digest('before_labels.json'),seed=410,
        score_sha256=digest(__file__),results=report),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
