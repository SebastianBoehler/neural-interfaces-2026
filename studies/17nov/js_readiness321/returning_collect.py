"""Collect all24 genuinely excluded new returning weights before any accuracy consumer."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/lustre/work/gehler/gfh527')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    records=[];masks={};seen=set()
    roots={'sinc':('ni26_js321_20261117_returning2','ni26_js321_20261117_returning_sinc_full1','outputs'),
           'cc30':('ni26_js321_20261117_returning_cc30_pilot1','ni26_js321_20261117_returning_cc30_full1','cc30')}
    for family,(pilot,rest,folder) in roots.items():
        for fold in ('P04_s23','P10_s23'):
            for seed in range(6):
                base=ROOT/(pilot if (fold,seed)==('P04_s23',0) else rest)
                path=base/folder/f'{fold}_s{seed}'/'proof.json';proof=json.loads(path.read_text())
                expected='new_genuine_held_session_weight_export_passed' if family=='sinc' else 'new_returning_CC30_source_export_passed'
                if proof['status']!=expected or (proof['fold'],proof['seed'])!=(fold,seed) or proof['historical_identity_claimed'] or proof['accuracy_computed']:
                    raise ValueError('wrong checkpoint scope/status')
                key=(fold,seed,family)
                if key in seen:raise ValueError('duplicate model')
                seen.add(key)
                source=proof.get('source_sha256',proof.get('source_manifest'))
                if source!=sha(base/'source.sha256'):raise ValueError('producer source changed')
                masks_here=tuple(proof[k] for k in ('train_rows_sha256','held_rows_sha256','train_count','held_count'))
                if fold in masks and masks[fold]!=masks_here:raise ValueError('source exclusions differ across arms/seeds')
                masks[fold]=masks_here
                for file,k in (('checkpoint.pt','checkpoint_sha256'),('trace_k20.pt' if family=='sinc' else 'trace_k2.pt','trace_sha256')):
                    if sha(path.parent/file)!=proof[k]:raise ValueError('checkpoint/export changed')
                records.append(dict(family=family,fold=fold,seed=seed,proof=str(path),proof_sha256=sha(path),
                                    checkpoint_sha256=proof['checkpoint_sha256'],trace_sha256=proof['trace_sha256']))
    if len(records)!=24:raise ValueError('incomplete returning matrix')
    Path('returning_complete.json').write_text(json.dumps(dict(status='all24_new_genuine_returning_weights_complete',
        records=records,masks=masks,accuracy_computed=False,historical_identity_claimed=False,
        whole_returning_bridge_still_required=True,source_sha256=sha('source.sha256')),indent=2))


if __name__=='__main__':main()
