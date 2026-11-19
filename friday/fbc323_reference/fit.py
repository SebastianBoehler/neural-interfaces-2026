"""Train one authenticated F26 fold without opening held labels."""
import argparse
from pathlib import Path

import numpy as np
import torch

from anchors import DATA_SHA, ORIGIN_SHA, PEOPLE, RUNTIME, row_sha, sha, split
from phase_source import F16_ACTOR, authenticated
from produce_distill import publish
from reference_map import HERE, KINDS, NATIVE_PARAMETERS, attach, probe
from source_targets import normalizer, raw_logits


F14 = HERE.parent / 'friday_s1c_training_20261118'
F16 = HERE.parent / 'friday_s1c_distill_20261118'


def run(held, seed, smoke):
    import braindecode
    import encoder312
    import model322
    from source_labels import read_training_labels

    path = F14 / 'proteus_windows.npz'
    if (sha(F16 / 'actor.sha256') != F16_ACTOR or sha(F14 / 'runtime.lock') != RUNTIME or
            sha(F14 / 'original/source323.sha256') != ORIGIN_SHA or sha(path) != DATA_SHA or
            Path(model322.__file__).resolve() != F14/'original/model322.py'):
        raise ValueError('F16 input, source, or runtime changed')
    device = torch.cuda.get_device_name()
    if 'H100' not in device or braindecode.__version__ != '1.8.1':
        raise ValueError('F26 requires exact Ferranti H100 runtime')
    torch.set_num_threads(2)
    with np.load(path, allow_pickle=False) as z:
        x, subject, visit = (z[k] for k in ('X','subject','session'))
    if x.shape != (2655,41,480) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('invalid pinned official EEG')
    rows, stream, people = split(subject, visit, held)
    ref, batches = authenticated(F16,F16,held,seed,smoke,rows,stream,people)
    labels = read_training_labels(path, rows, len(x))
    x,y,scale,mu,sd,weights,tape = model322.setup(dict(X=x[rows],y=labels,stream=stream),
                                                  np.arange(len(rows)),'cuda',epochs=ref['epochs'])
    flat = [b for epoch in batches for b in epoch]
    if scale != ref['scale'] or any(not np.array_equal(a,b) for a,b in zip(flat,(r for e in tape for r in e),strict=True)):
        raise ValueError('original source scale or schedule changed')
    sample = x[flat[0][:16]]
    tag = f"{'smoke_' if smoke else ''}{seed}_{held}"
    common = dict(held=held,seed=seed,epochs=ref['epochs'],scale=scale,data=DATA_SHA,origin=ORIGIN_SHA,
                  runtime=RUNTIME,weights_origin=ref['checkpoint'],initial=ref['initial'],batches=ref['batches'],
                  source_rows=row_sha(rows),source_people=people,actor=sha(HERE/'actor.sha256'),
                  held_labels_opened=False, device=device,torch=str(torch.__version__),braindecode=braindecode.__version__,
                  diagnostic_rows=row_sha(rows[flat[0][:16]]),parameters=NATIVE_PARAMETERS)
    states = set()
    for kind in KINDS:
        model = model322.Factor('session',False,'cuda')
        normalizer(model)
        if encoder312.fingerprint(model) != ref['initial']:
            raise ValueError('source initialization differs')
        calls = [0]
        handle = attach(model, kind, calls)
        zero = []
        gradient = model.task.weight.register_hook(lambda g: zero.append(g[:, -287:].count_nonzero()))
        try:
            learned = model322.train(model,x,y,mu,sd,weights,batches)
        finally:
            gradient.remove()
        if (calls[0] != (0 if kind=='hard' else ref['steps']) or len(zero) != ref['steps'] or
                learned['steps'] != ref['steps'] or torch.stack(zero).sum().item()):
            raise ValueError('native source update count or feature-zero checks failed')
        proof = probe(sample,kind)
        with torch.no_grad():
            model.eval()
            if not raw_logits(model,sample).isfinite().all():
                raise ValueError('trained native logits are nonfinite')
        if handle is not None:
            handle.remove()
        state = encoder312.fingerprint(model)
        if state == ref['initial'] or state in states or (kind=='hard' and state != ref['state']):
            raise ValueError('source arms are duplicate or native hard state differs')
        states.add(state)
        output = HERE/f'{kind}_{tag}.pt'
        publish(output,model,dict(common,**learned,kind=kind,state=state,fit_map_calls=0 if kind=='hard' else ref['steps'],
                                  q_gradients_nonzero=0,**proof))
        if kind=='hard' and sha(output) != ref['checkpoint']:
            raise ValueError('hard checkpoint is not byte exact')
    print(f'PASS F26 paired source {tag}',flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('held', choices=PEOPLE)
    cli.add_argument('seed', type=int, choices=(0,1))
    cli.add_argument('--smoke', action='store_true')
    args = cli.parse_args()
    run(args.held,args.seed,args.smoke)
