"""Replace the operator at its actual location, replay the full original waveform decoder."""
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

TRAIN=Path('/mnt/lustre/work/gehler/gfh527/ni26_onebn381_v2_20261118')
sys.path.insert(0,str(TRAIN))
from onebn import LearnedBN,joint,preactivation,source
TRAIN_SHA='bcbb7ea6301a74c0dbb82df34dce75f51fdc7bcf16147caa12bdde568538c215'
ANCHOR=TRAIN.parent/'ni26_joint373_anchor_v2_20261117'
ANCHOR_SHA='96bcd7e7310236dfc01a7c64369d92f753dfc8dbf851fde276f961e386803141'


def owners(rows):
    joint.check_manifest(TRAIN,TRAIN_SHA)
    fixed=json.loads((ANCHOR/'before_labels.json').read_text())
    joint.pin(ANCHOR/'before_labels.json',ANCHOR_SHA)
    joint.pin(ANCHOR/'outputs.npz',fixed['output_sha'])
    records={arm:json.loads((TRAIN/f'train_{arm}.json').read_text()) for arm in ('C','D')}
    assert records['C']['episodes']==records['D']['episodes']
    assert records['C']['checks']['initial_state']==records['D']['checks']['initial_state']
    for m in records.values():
        assert m['train_rows']==joint.digest(rows) and m['source_sha']==TRAIN_SHA
        assert m['held_labels_opened'] is False and not m['smoke'] and tuple(m['train_people'])==joint.FIT_PEOPLE
        for name,sha in m['checkpoints'].items():
            joint.pin(TRAIN/(name+'.pt'),sha)
    return records


@torch.no_grad()
def single_steps(layer,z,mode,donor):
    total=torch.zeros_like(layer.encode(z[:1])[0])
    out=[]
    items=layer.encode(z)
    for i,item in enumerate(items):
        count=1 if mode=='blank' else i+1
        prior=(torch.zeros_like(total) if mode=='blank' else total)
        mean=(prior+item)/count
        mix=count/(count+20)
        inp=torch.cat((mean.asinh(),torch.full_like(mean[:,:1],mix)),-1)
        a,b=layer.read(inp).unbind(-1)
        result=layer.transform(z[i:i+1],(3.5*(a/3.5).tanh()*mix)[None],(4*(b/4).tanh()*mix)[None])
        out.append(result)
        total=total+(donor[i%len(donor)] if mode=='donor' else item)
    return torch.cat(out)


@torch.no_grad()
def run(person):
    start=time.monotonic()
    d,people,rows,runtime=source()
    fits=owners(rows)
    proof=joint.verify()
    streams,held,meta,people2=joint.banks(proof)
    assert np.array_equal(people,people2) and person in held
    test=held[person]
    assert not np.intersect1d(test,rows).size and len(set(d['stream'][test]))==1
    donor_rows=np.flatnonzero((people=='P04') & (d['session'].astype(str)=='1'))
    assert len(donor_rows)==189 and np.isin(donor_rows,rows).all()
    original=runtime.sbns()[0]
    assert runtime.net.spatial_conv[1] is original
    perms=joint.permutations(len(people))
    students={}
    donor_bank={}
    # Compute the exact fit-specific feature reconstruction for the source donor.
    donor_input=torch.from_numpy(runtime._aligned(donor_rows,True))
    h=preactivation(runtime.net,donor_input)
    for kind in ('C','D','DC'):
        layer=LearnedBN(original).eval()
        layer.load_state_dict(torch.load(TRAIN/(kind+'.pt'),map_location='cpu',weights_only=True))
        students[kind]=layer
        donor_bank[kind]=layer.encode(layer.standardized(h))
    outputs={}
    checks={'a6':0.,'population':0.,'step':0.,'prefix':0.,'fixed_set':0.}
    with np.load(ANCHOR/'outputs.npz') as archive:
        for order in joint.ORDERS:
            positions=streams[person][order][2]
            rr=test[positions]
            assert np.array_equal(rr,perms[order][np.isin(perms[order],test)])
            x=torch.roll(torch.from_numpy(runtime._aligned(rr,True)),0,-1)
            undo=np.argsort(positions)
            runtime.net.spatial_conv[1]=original
            for arm,k in [('A6',20.),('population',None)]:
                original.k=k
                v=runtime.net(x).double().softmax(1).numpy()[undo]
                ref=archive[person+'|'+order+'|'+arm]
                err=float(np.max(np.abs(v-ref)))
                checks['a6' if arm=='A6' else arm]=max(checks['a6' if arm=='A6' else arm],err)
                assert err<=1e-6 and np.array_equal(v.argmax(1),ref.argmax(1)),(person,order,arm,err)
                outputs[arm+'|'+order]=v
            for ref in ('frozen','JS'):
                outputs[ref+'|'+order]=archive[person+'|'+order+'|'+ref]
            donor_order=np.searchsorted(donor_rows,perms[order][np.isin(perms[order],donor_rows)])
            for kind,layer in students.items():
                runtime.net.spatial_conv[1]=layer
                layer.donor=donor_bank[kind][donor_order]
                for mode in ('real','blank','donor'):
                    layer.mode=mode
                    logits=runtime.net(x)
                    assert torch.isfinite(logits).all()
                    outputs[kind+'_'+mode+'|'+order]=logits.double().softmax(1).numpy()[undo]
                    if order==joint.ORDERS[0]:
                        short=x[:8]
                        local=preactivation(runtime.net,short)
                        full=layer(local)
                        single=single_steps(layer,layer.standardized(local),mode,layer.donor)
                        err=float((full-single).abs().max())
                        checks['step']=max(checks['step'],err)
                        torch.testing.assert_close(single,full,atol=2e-4,rtol=1e-5)
                        zz=runtime.net(short)
                        err=float((logits[:8]-zz).abs().max())
                        checks['prefix']=max(checks['prefix'],err)
                        torch.testing.assert_close(zz,logits[:8],atol=3e-4,rtol=1e-5)
                        assert torch.equal(zz,runtime.net(short))
                        if mode=='real':
                            shuffled=torch.cat((short[:-1].flip(0),short[-1:]))
                            other=runtime.net(shuffled)[-1]
                            checks['fixed_set']=max(checks['fixed_set'],float((other-zz[-1]).abs().max()))
                            torch.testing.assert_close(zz[-1],other,atol=3e-4,rtol=1e-5)
    assert len(outputs)==13*8
    np.savez_compressed(person+'.npz',index=test,**outputs)
    report=dict(held=person,train_rows=joint.digest(rows),test_rows=joint.digest(test),train_people=joint.FIT_PEOPLE,
                train_source=TRAIN_SHA,actor_source=joint.sha(Path('source.sha256')),held_labels_opened=False,
                checks=checks,outputs_sha256=joint.sha(Path(person+'.npz')),seconds=time.monotonic()-start)
    Path(person+'.json').write_text(json.dumps(report,indent=2))
    print(person,report,flush=True)


if __name__=='__main__':
    run(sys.argv[1])
