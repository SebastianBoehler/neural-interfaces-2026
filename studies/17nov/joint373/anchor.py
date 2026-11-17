"""Retain the exact strong causal current logits while training the waveform correction."""
import copy
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch

ROOT = Path('/mnt/lustre/work/gehler/gfh527/ni26_joint373_v2_20261117')
spec = importlib.util.spec_from_file_location('joint373', ROOT/'pilot.py')
joint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(joint)
joint.check_manifest(ROOT, '5269d87eb3c10b5d6ead65745fd126915b101ba7b1aebf524ed9356a41513dab')


class Anchored(joint.Joint):
    def __init__(self, body, causal):
        super().__init__(body)
        self.causal = causal.requires_grad_(False)
        self.population = copy.deepcopy(body).requires_grad_(False)

    def encode(self, x):
        current, item = super().encode(x)
        with torch.no_grad():
            fixed = self.causal(torch.roll(x, 0, -1))
            population = self.population(x)
        return fixed + .5 * (current - population).tanh(), item


@torch.no_grad()
def source_guards(model, x):
    z, items = model.encode(x)
    rev = [*range(len(x)-2, -1, -1), len(x)-1]
    zr, ir = model.encode(x[rev])
    direct = model.predict(z, items, True)
    torch.testing.assert_close(direct[-1], model.predict(zr, ir, True)[-1], atol=2e-4, rtol=1e-5)
    for count in (1, 6):
        zz, ii = model.encode(x[:count])
        torch.testing.assert_close(direct[:count], model.predict(zz, ii, True), atol=2e-4, rtol=1e-5)
    assert (direct[-1] - model.predict(*model.encode(x[-1:]), True)[0]).abs().max() > 1e-6


def main():
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(373)
    primary = json.loads((ROOT/'before_labels.json').read_text())
    joint.pin(ROOT/'outputs.npz', primary['outputs_sha'])
    with np.load(ROOT/'outputs.npz') as bank:
        outputs = {tuple(k.split('|')):bank[k] for k in bank.files if k.split('|')[-1] in ('frozen','JS','A6')}
    proof = joint.verify()
    streams, held, meta, people = joint.banks(proof)
    data, _, _, people2 = joint.load()
    assert np.array_equal(people, people2) and 'y' not in data
    rows = np.flatnonzero(np.isin(people, joint.FIT_PEOPLE))
    assert joint.digest(rows) == primary['training_rows']
    runtime = joint.restore(data, rows, joint.verify_body())
    runtime.set_k(20.)
    causal = copy.deepcopy(runtime.net)
    runtime.set_k(None)
    init = Anchored(runtime.net, causal).eval()
    causal_sha = joint.state_sha(causal)
    x = torch.from_numpy(runtime._aligned(rows, True))
    y = joint.labels(rows, len(people))
    batches, weights = joint.episodes(rows, data, y)
    y = torch.from_numpy(y.astype(np.int64))
    assert joint.digest(np.stack(batches)) == primary['traces']['prefix']['batches']
    source_guards(init, x[:9])
    with torch.no_grad():
        z, _ = init.encode(x[:9])
        assert torch.equal(z, causal(torch.roll(x[:9], 0, -1)))
    traces, maxerr = {}, 0.
    for arm in ('prefix','current'):
        prefix = arm == 'prefix'
        joint.fit(init,x,y,batches[:1],weights,prefix)
        net, traces[arm] = joint.fit(init,x,y,batches,weights,prefix)
        source_guards(net,x[:9])
        joint.guards(net,x[:9])
        assert joint.state_sha(net.causal) == joint.state_sha(net.population) == causal_sha
        for person, r in held.items():
            assert len(set(meta['stream'][r])) == 1 and not np.intersect1d(r,rows).size
            for order in joint.ORDERS:
                positions = streams[person][order][2]
                with torch.no_grad():
                    delivered = torch.from_numpy(runtime._aligned(r[positions], True))
                    fixed = net.causal(torch.roll(delivered, 0, -1)).double().softmax(1).numpy()[np.argsort(positions)]
                    ref = outputs[person,order,'A6']
                    err = np.max(np.abs(ref-fixed))
                    maxerr = max(maxerr, float(err))
                    assert err <= 1e-6 and np.array_equal(ref.argmax(1),fixed.argmax(1)), (person,order,err)
                    z, items = net.encode(delivered)
                    raw = net.population(delivered).double().softmax(1)
                    outputs[person,order,'population'] = raw.numpy()[np.argsort(positions)]
                    p = net.predict(z, items, prefix).double().softmax(1).numpy()
                    assert np.isfinite(p).all()
                    outputs[person,order,arm] = p[np.argsort(positions)]
        print('arm complete',arm,flush=True)
    assert len(outputs) == 4*8*6
    np.savez_compressed('outputs.npz', **{'|'.join(k):v for k,v in outputs.items()})
    before = dict(source_sha=joint.sha(Path('source.sha256')), held_labels_opened=False,
                  output_sha=joint.sha(Path('outputs.npz')), training_rows=joint.digest(rows), traces=traces,
                  frozen_causal_anchor_max_error=maxerr, snapshot=causal_sha)
    Path('before_labels.json').write_text(json.dumps(before,indent=2))
    scorer = joint.official()
    joint.pin(Path(scorer.__file__), joint.METRIC_SHA)
    result = {}
    for ref in ('A6','population','frozen','JS'):
        matrix = {**outputs, **{(p,o,'A6'):outputs[p,o,ref] for p in joint.COHORT for o in joint.ORDERS}}
        result[ref] = joint.score(scorer,joint.labels,meta,people,joint.COHORT,joint.ORDERS,matrix)
    matrix = {**outputs, **{(p,o,'prefix'):outputs[p,o,'A6'] for p in joint.COHORT for o in joint.ORDERS},
              **{(p,o,'A6'):outputs[p,o,'population'] for p in joint.COHORT for o in joint.ORDERS}}
    result['causal_vs_population'] = joint.score(scorer,joint.labels,meta,people,joint.COHORT,joint.ORDERS,matrix)
    result['seconds'] = time.monotonic()-start
    Path('result.json').write_text(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
