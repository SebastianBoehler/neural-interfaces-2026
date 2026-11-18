"""Replay frozen §382 operators and intervene solely on earlier writes."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from intervene import variants
from source import BUNDLE, CHECKPOINTS, ROWS

WORK = Path('/mnt/lustre/work/gehler/gfh527')
sys.path.insert(0,str(WORK/'ni26_conflict388_v1_20261118'))
from actor388 import ARCH, init, joint, prior
from onebn import Tail

MODES = ('real','donor','blank')
KINDS = ('unchanged','semantic','wrong')
ARMS = ('population','A6','frozen') + tuple(f'{k}_{m}' for k in KINDS for m in MODES)
PROTO_SHA = '1d7108832dd70d5984a394cee66c1592966b7db4b00e52a2f4cb09fc2e7f7fb9'


def affine_logits(layer, tail, z, items, mode, writes):
    args = layer.affine(items,'blank') if mode == 'blank' else layer.affine(items,'donor',writes)
    return tail(layer.transform(z,*args))


def parity(actual, archive, atol=1e-6):
    deviation = float(abs(actual-archive).max())
    assert deviation < atol and np.array_equal(actual.argmax(1),archive.argmax(1)), deviation
    return deviation


@torch.no_grad()
def run(person):
    src = json.loads(Path('source.json').read_text())
    assert src['bundle']==BUNDLE and src['rows']==ROWS and tuple(src['people'])==joint.FIT_PEOPLE
    assert src['source']=='05fb9fc4136b0d96c0e71ddd811f362a2c8ef0efe910c6f3455f21f024dcc1da'
    joint.pin(Path('prototypes.npz'),PROTO_SHA)
    assert src['output']==PROTO_SHA
    d, people, rows, runtime = prior.source()
    prior.owners(rows)
    streams, held, _, people2 = joint.banks(joint.verify())
    test = held[person]
    assert np.array_equal(people,people2) and np.array_equal(test,np.flatnonzero(people==person))
    assert joint.digest(rows)==ROWS and not np.intersect1d(test,rows).size
    assert len(np.unique(d['stream'][test]))==1 and not runtime.align and not runtime.crops
    donor = np.flatnonzero((people=='P04') & (d['session'].astype(str)=='1'))
    assert len(donor)==189 and np.isin(donor,rows).all()
    donor_x = torch.from_numpy(runtime._aligned(donor,True))
    pop = runtime.net
    pop.spatial_conv[1].k = None
    donor_p = pop(donor_x).softmax(1)
    perms = joint.permutations(len(people))
    bank, diagnostics = {}, {}
    with np.load('prototypes.npz') as templates:
        for seed, (root, digest) in ARCH.items():
            joint.pin(root/'before_labels.json',digest)
            proof = json.loads((root/'before_labels.json').read_text())['actors'][person]
            assert proof['train_rows']==ROWS and proof['test_rows']==joint.digest(test)
            joint.pin(root/(person+'.npz'),proof['outputs_sha256'])
            weights = Path(json.loads((root/'fit_lock.json').read_text())['root'])/f'J_{seed}.pt'
            joint.pin(weights,CHECKPOINTS[seed])
            net = init(runtime.net,'J')
            net.load_state_dict(torch.load(weights,weights_only=True,map_location='cpu'))
            layer = net.spatial_conv[1]
            tail = Tail(net).eval()
            proto = torch.from_numpy(templates[str(seed)])
            def encode(x):
                z = layer.standardized(prior.preactivation(net,x))
                return z,layer.encode(z)
            _, probe = encode(torch.from_numpy(runtime._aligned(rows[:8],True)))
            torch.testing.assert_close(probe,torch.from_numpy(templates[f'probe{seed}']),atol=5e-5,rtol=1e-5)
            _, donor_items = encode(donor_x)
            donor_variants, donor_size = variants(donor_items,donor_p,proto)
            with np.load(root/(person+'.npz')) as arch:
                assert np.array_equal(arch['index'],test)
                for order in joint.ORDERS:
                    pos = streams[person][order][2]
                    rr = test[pos]
                    assert np.array_equal(rr,perms[order][np.isin(perms[order],test)])
                    undo = np.argsort(pos)
                    x = torch.from_numpy(runtime._aligned(rr,True))
                    for name, k in [('population',None),('A6',20.)]:
                        pop.spatial_conv[1].k=k
                        logits=pop(x)
                        probs=logits.double().softmax(1)
                        parity(probs.numpy()[undo],arch[name+'|'+order])
                        if name=='population':
                            posterior=logits.softmax(1)
                        bank[f'{seed}|{name}|{order}']=probs.argmax(1).numpy()[undo].astype(np.int8)
                    bank[f'{seed}|frozen|{order}']=arch['frozen|'+order].argmax(1).astype(np.int8)
                    z, items = encode(x)
                    own, size = variants(items,posterior,proto)
                    ids=np.searchsorted(donor,perms[order][np.isin(perms[order],donor)])
                    assert len(ids)==len(donor)
                    outputs={}
                    for mode in MODES:
                        for kind in KINDS:
                            writes = donor_variants[kind][ids] if mode=='donor' else own[kind]
                            v=affine_logits(layer,tail,z,items,mode,writes)
                            assert torch.isfinite(v).all()
                            outputs[kind,mode]=v
                            if kind=='unchanged':
                                parity(v.double().softmax(1).numpy()[undo],arch['J_'+mode+'|'+order])
                            assert torch.equal(v[0],outputs['unchanged',mode][0])
                            if mode=='blank':
                                assert torch.equal(v,outputs['unchanged',mode])
                            if order==joint.ORDERS[0]:
                                short=affine_logits(layer,tail,z[:8],items[:8],mode,writes[:8])
                                torch.testing.assert_close(short,v[:8],atol=3e-4,rtol=1e-5)
                            bank[f'{seed}|{kind}_{mode}|{order}']=v.argmax(1).numpy()[undo].astype(np.int8)
                    scale = {}
                    for kind in ('semantic','wrong'):
                        for mode in ('real','donor'):
                            diff=(outputs[kind,mode]-outputs['unchanged',mode]).abs()
                            assert diff[1:].max()>1e-5
                            scale[kind+'_'+mode]=float(diff.mean())
                    diagnostics[f'{seed}|{order}']=dict(write_relative=size,donor_relative=donor_size,
                                                          logit_mean_abs=scale)
    path=Path(person+'.npz')
    np.savez_compressed(path,index=test,**bank)
    rec=dict(held=person,train=ROWS,test=joint.digest(test),output=joint.sha(path),
             source=joint.sha(Path('actor.sha256')),prototypes=PROTO_SHA, activity=diagnostics,
             held_labels_opened=False)
    Path(person+'.json').write_text(json.dumps(rec,indent=2))


if __name__ == '__main__':
    run(sys.argv[1])
