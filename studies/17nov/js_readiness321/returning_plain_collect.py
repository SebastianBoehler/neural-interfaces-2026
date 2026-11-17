"""Owning all28 genuine plain-family source/asset collector; no held score."""
import json
from pathlib import Path

from returning_train import sha

ROOT=Path('/mnt/lustre/work/gehler/gfh527')
FAMILIES={'shallow_ea':5,'shallow_raw':5,'fbcnet_ea_tf':2,'fbcnet_raw_tf':2}


def asset(cfg,fold,seed):
    if (fold,seed)==('P04_s23',0):
        work='returning_plain_recover1' if cfg=='shallow_ea' else 'returning_plain_pilot2'
    else:work='returning_plain_full1'
    return ROOT/f'ni26_js321_20261117_{work}'/'plain'/f'{cfg}_{fold}_s{seed}'


def checked(cfg,fold,seed):
    path=asset(cfg,fold,seed);p=json.loads((path/'proof.json').read_text());n=2278 if fold=='P04_s23' else 2270
    if (p['status'],p['family'],p['fold'],p['seed'],p['train_count'],p['held_count'])!=('new_source_plain_family_export_passed_NOT_accuracy',cfg,fold,seed,n,2655-n):raise ValueError('source proof mismatch')
    if p['accuracy_computed'] or p['historical_identity_claimed'] or not p['immutable_buffers'] or not p['samepath_interleaved_bitwise']:raise ValueError('source/export scope failed')
    if max(p['export_errors'].values())>1e-4:raise ValueError('own export guard failed')
    for name in ('checkpoint','trace'):
        file=path/('checkpoint.pt' if name=='checkpoint' else 'trace.pt')
        if sha(file)!=p[name+'_sha256']:raise ValueError('source weights/trace changed')
    return dict(family=cfg,fold=fold,seed=seed,path=str(path),proof_sha256=sha(path/'proof.json'),checkpoint_sha256=p['checkpoint_sha256'],trace_sha256=p['trace_sha256'])


if __name__=='__main__':
    records=[checked(cfg,fold,seed) for cfg,n in FAMILIES.items() for fold in ('P04_s23','P10_s23') for seed in range(n)]
    Path('plain_complete.json').write_text(json.dumps(dict(status='all28_new_genuine_source_plain_weights_complete_NOT_accuracy',assets=records,accuracy_computed=False,source_sha256=sha('source.sha256')),indent=2))
