"""Fixed genuine returning-fold asset map; no missing-file substitutions."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/lustre/work/gehler/gfh527')
KINDS={'sinc':20.,'cc30':2.,'fb':2.,'fb5':5.,'fbl':5.,'fbraw':20.}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def asset(kind,fold,seed):
    if kind not in KINDS or fold not in ('P04_s23','P10_s23') or seed not in range(6):raise ValueError('registered source required')
    family='fb' if kind=='fb5' else kind;pilot=(fold,seed)==('P04_s23',0)
    if family=='sinc':
        base=ROOT/('ni26_js321_20261117_returning2' if pilot else 'ni26_js321_20261117_returning_sinc_full1');folder=base/'outputs'/f'{fold}_s{seed}'
    elif family=='cc30':
        base=ROOT/('ni26_js321_20261117_returning_cc30_pilot1' if pilot else 'ni26_js321_20261117_returning_cc30_full1');folder=base/'cc30'/f'{fold}_s{seed}'
    else:
        base=ROOT/('ni26_js321_20261117_returning_other_pilot1' if pilot else 'ni26_js321_20261117_returning_other_full1');folder=base/'other'/f'{family}_{fold}_s{seed}'
    proof=json.loads((folder/'proof.json').read_text())
    expected_status={'sinc':'new_genuine_held_session_weight_export_passed','cc30':'new_returning_CC30_source_export_passed'}.get(family,'new_genuine_other_family_source_export_passed')
    if proof['status']!=expected_status:raise ValueError('owning source/export proof absent')
    if (proof['fold'],proof['seed'])!=(fold,seed) or proof['historical_identity_claimed'] or proof['accuracy_computed']:
        raise ValueError('source scope differs')
    if proof.get('source_sha256',proof.get('source_manifest'))!=sha(base/'source.sha256'):raise ValueError('producer source changed')
    ckpt=folder/'checkpoint.pt';trace=folder/f'trace_k{KINDS[kind]:g}.pt'
    expected=proof['trace_sha256'] if family in ('sinc','cc30') else proof['exports'][str(KINDS[kind])]['sha256']
    if sha(ckpt)!=proof['checkpoint_sha256'] or sha(trace)!=expected:raise ValueError('asset changed')
    return ckpt,trace,proof
