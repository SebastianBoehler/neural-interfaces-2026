"""Independently replay each F26 arm under both native normalization modes."""
import argparse
import json

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from held_distill import replay
from phase_source import F16_BANK, authenticated
from reference_map import ARMS, HERE, KINDS, MODES, NATIVE_PARAMETERS, attach, probe
from replay_f13 import ORDER_SHA, ORDERS, origin_report, raw_inputs
from source_targets import normalizer


ACTOR = 'b93711fe591771ffdd51dff50f4b07a7a21d3e1500635ebdfaecfabde818c2d2'
F16 = HERE.parent / 'ni26_friday_s1c_distill_20261118'


def paired(held, seed, smoke, subject, visit):
    if sha(HERE/'actor.sha256') != ACTOR or sha(F16/'before_labels.sha256') != F16_BANK:
        raise ValueError('F26 or F16 blind ancestry differs')
    rows, stream, people = split(subject,visit,held)
    ref, batches = authenticated(F16,F16/'checkpoints',held,seed,smoke,rows,stream,people)
    expected = dict(held=held,seed=seed,data=DATA_SHA,origin=ORIGIN_SHA,runtime=RUNTIME,
                    actor=ACTOR,source_rows=row_sha(rows),source_people=people,epochs=ref['epochs'],
                    steps=ref['steps'],scale=ref['scale'],initial=ref['initial'],weights_origin=ref['checkpoint'],
                    diagnostic_rows=row_sha(rows[batches[0][0][:16]]),parameters=NATIVE_PARAMETERS,
                    batches=ref['batches'],held_labels_opened=False,q_gradients_nonzero=0,
                    torch=ref['torch'],braindecode=ref['braindecode'])
    result = {}
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    for kind in KINDS:
        path = HERE/'checkpoints'/f'{kind}_{tag}.pt'
        proof = json.loads(path.with_suffix('.json').read_text())
        pinned = dict(expected,kind=kind,checkpoint=sha(path),fit_map_calls=0 if kind=='hard' else ref['steps'])
        if any(proof[k] != v for k,v in pinned.items()) or 'H100' not in proof['device']:
            raise ValueError(f'F26 source recipe changed for {kind}/{tag}')
        values = [proof[k] for k in ('first_loss','last_loss','max_gradient','delta','orthogonal','norm_error','common_offset')]
        if not np.isfinite(values).all() or proof['max_gradient'] <= 0 or proof['state'] == proof['initial']:
            raise ValueError('invalid optimizer or reference receipt')
        if kind=='hard':
            if proof['checkpoint'] != ref['checkpoint'] or proof['state'] != ref['state'] or any(values[3:]):
                raise ValueError('F16 hard state not exact')
        elif not (proof['delta'] > 1e-5 and 0 <= proof['norm_error'] <= 2e-6 and
                  0 <= proof['orthogonal'] <= 2e-6 and
                  (proof['common_offset'] <= 5e-6 if kind=='car' else proof['common_offset'] >= .02)):
            raise ValueError('learned F26 source map proof failed')
        result[kind] = (path,proof)
    if len({p['state'] for _,p in result.values()}) != len(KINDS):
        raise ValueError('duplicate F26 learned states')
    return result


def run(held, seed, smoke):
    import encoder312
    import model322

    torch.set_num_threads(4)
    origin_report()
    x, subject, person, stream, perms = raw_inputs()
    rows = np.flatnonzero(person==held)
    visit = np.asarray([r.rsplit('|',1)[-1] for r in stream])
    decisions, evidence = {}, {}
    for kind,(path,meta) in paired(held,seed,smoke,subject,visit).items():
        model322.SEED = seed
        model = model322.Factor('session',False,'cpu')
        weights = torch.load(path,map_location='cpu',weights_only=True)
        if any(weights[k] != meta[k] for k in ('kind','held','seed','scale')):
            raise ValueError('F26 checkpoint identity differs')
        model.load_state_dict(weights['state'],strict=True)
        model.eval()
        if encoder312.fingerprint(model) != meta['state']:
            raise ValueError('F26 frozen checkpoint state differs')
        calls = [0]
        hook = attach(model,kind,calls)
        guard = probe(torch.from_numpy(x[rows[:16]] / meta['scale']),kind)
        variants, proof = replay(model,normalizer(model),x,stream,rows,perms,meta['scale'])
        if (kind=='hard' and calls[0]) or (kind!='hard' and calls[0] <= proof['resets']):
            raise ValueError('F26 active held map count failed')
        if hook is not None:
            hook.remove()
        for name, bank in zip(MODES,variants,strict=True):
            decisions[f'{kind}_{name}'] = bank
        evidence[kind] = dict(proof,reference=guard,reference_calls=calls[0],checkpoint=meta['checkpoint'],
                              receipt=sha(path.with_suffix('.json')),state=meta['state'])
    if tuple(decisions) != ARMS or not any(x['clips'] for x in evidence.values()):
        raise ValueError('F26 held bank or bounded work missing')
    receipt = dict(held=held,seed=seed,actor=ACTOR,scorer=sha(HERE/'score.sha256'),data=DATA_SHA,
                   origin=ORIGIN_SHA,orders=list(ORDERS),order_sha=ORDER_SHA,held_labels_opened=False,evidence=evidence)
    if not smoke:
        prior = F16/'folds'/f'{held}_s{seed}.npz'
        with np.load(prior,allow_pickle=False) as z:
            if not np.array_equal(z['index'],rows) or any(not np.array_equal(z[f'hard_{m}'],decisions[f'hard_{m}']) for m in MODES):
                raise ValueError('F26 hard held decisions differ from original F16')
        path = HERE/'folds'/prior.name
        with path.open('xb') as out:
            np.savez_compressed(out,index=rows,metadata=json.dumps(receipt,sort_keys=True),**decisions)
        path.with_suffix('.sha256').write_text(sha(path)+'\n')
    print(json.dumps(receipt,sort_keys=True),flush=True)
    print(f'PASS F26 held {held} seed {seed}',flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('held',choices=PEOPLE)
    cli.add_argument('seed',type=int,choices=(0,1))
    cli.add_argument('--smoke',action='store_true')
    args = cli.parse_args()
    run(args.held,args.seed,args.smoke)
