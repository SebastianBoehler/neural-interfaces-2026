"""Own all576 native-source bank files before any returning accuracy consumer."""
import json
from pathlib import Path

import numpy as np

from returning_assets import ROOT,KINDS,asset,sha
from returning_bank import ORDERS


def main():
    records=[]
    for fold in ('P04_s23','P10_s23'):
        for seed in range(6):
            pilot=(fold,seed)==('P04_s23',0)
            base=ROOT/('ni26_js321_20261117_returning_bank_native1' if pilot else 'ni26_js321_20261117_returning_bank_full1')
            folder=base/('banks' if pilot else f'banks_{fold}_s{seed}')/f'{fold}_s{seed}'
            proof=json.loads((folder/'proof.json').read_text())
            if proof['status']!='native_returning_B1_bank_guard_passed' or (proof['fold'],proof['seed'])!=(fold,seed):raise ValueError('native cell proof missing')
            if proof['source_sha256']!=sha(base/'source.sha256') or proof['torch']!='2.14.1+cu130' or proof['accuracy_computed']:raise ValueError('source/executor/scope differs')
            if proof['cudnn_tf32'] or proof['matmul_tf32'] or not proof['old_batch_failure_preserved']:raise ValueError('protocol differs')
            if len(proof['checks'])!=96 or set(proof['files'])!={f'{o}_{k}.npz' for o in ORDERS for k in KINDS}:raise ValueError('incomplete order/session/kind coverage')
            for c in proof['checks']:
                if c['self_repeat_max']>1e-8 or c['interleaved_max']>1e-8 or not c['immutable_unchanged'] or not c['reset_zero']:raise ValueError('native semantic check failed')
            for name,digest in proof['files'].items():
                path=folder/name
                if sha(path)!=digest:raise ValueError('native bank changed')
                with np.load(path,allow_pickle=False) as z:
                    meta=json.loads(str(z['metadata']));p=z['proba'];idx=z['index']
                    ckpt,trace,_=asset(meta['kind'],fold,seed)
                    if meta['asset']!={'checkpoint_sha256':sha(ckpt),'trace_sha256':sha(trace)}:raise ValueError('bank source weight differs')
                    if meta['accuracy_computed'] or meta['executor']!='nativeSIF_B1_TF32off' or not np.all(np.diff(idx)>0):raise ValueError('bank scope/index differs')
                    if p.shape!=(len(idx),3) or len(idx)!=(377 if fold=='P04_s23' else 385) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1.):raise ValueError('invalid bank matrix')
                records.append(dict(fold=fold,seed=seed,name=name,path=str(path),sha256=digest))
    if len(records)!=576:raise ValueError('bank matrix incomplete')
    Path('native_banks_complete.json').write_text(json.dumps(dict(status='all576_native_returning_banks_complete_NO_ACCURACY',
        records=records,source_sha256=sha('source.sha256'),whole_returning_bridge_still_required=True,
        old_batch_failure_preserved=True,accuracy_computed=False),indent=2))


if __name__=='__main__':main()
