"""Discriminate session-specific information from a generic shrinkage correction."""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch

PRIMARY = Path('/mnt/lustre/work/gehler/gfh527/ni26_joint373_v2_20261117')
# Give the current implementation a distinct module name. It owns legacy module imports.
spec = importlib.util.spec_from_file_location('joint373', PRIMARY/'pilot.py')
joint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(joint)
joint.check_manifest(PRIMARY, '5269d87eb3c10b5d6ead65745fd126915b101ba7b1aebf524ed9356a41513dab')


@torch.no_grad()
def encoded(model, x):
    pairs = [model.encode(chunk) for chunk in x.split(32)]
    return tuple(torch.cat(components) for components in zip(*pairs))


def external(model, logits, items, contexts):
    # Duplicate only the final two operators so a foreign context can be injected.
    features = torch.cat((items, contexts), 1)
    return logits + model.gate(features).sigmoid() * model.move(features).tanh()


def prior(items):
    sums = torch.cat((torch.zeros_like(items[:1]), items[:-1].cumsum(0)))
    return sums / torch.arange(len(items), dtype=items.dtype).clamp_min(1)[:, None]


def main():
    start = time.monotonic()
    torch.set_num_threads(6)
    torch.manual_seed(374)
    record = json.loads((PRIMARY/'before_labels.json').read_text())
    joint.pin(PRIMARY/'outputs.npz', record['outputs_sha'])
    with np.load(PRIMARY/'outputs.npz') as bank:
        outputs = {tuple(key.split('|')):bank[key] for key in bank.files
                   if key.split('|')[-1] in ('frozen', 'JS', 'A6')}
    proof = joint.verify()
    streams, held, meta, people = joint.banks(proof)
    data, _, _, people2 = joint.load()
    assert np.array_equal(people, people2) and 'y' not in data
    rows = np.flatnonzero(np.isin(people, joint.FIT_PEOPLE))
    assert joint.digest(rows) == record['training_rows']
    runtime = joint.restore(data, rows, joint.verify_body())
    runtime.set_k(None)
    initial = joint.Joint(runtime.net).eval()
    x = torch.from_numpy(runtime._aligned(rows, True))
    y = joint.labels(rows, len(people))
    batches, weights = joint.episodes(rows, data, y)
    assert joint.digest(np.stack(batches)) == record['traces']['prefix']['batches']
    batches = [batches[i] for i in np.random.default_rng(374).permutation(64)]
    y = torch.from_numpy(y.astype(np.int64))
    perm = joint.permutations(len(people))
    source_streams = meta['stream'][rows]
    donor = np.flatnonzero(people == 'P05')
    assert len(donor) == 190 and np.isin(donor, rows).all() and len(set(meta['stream'][donor])) == 1
    traces = {}
    for arm in ('current', 'prefix'):
        use_prefix = arm == 'prefix'
        joint.fit(initial, x, y, batches[:1], weights, use_prefix)
        model, traces[arm] = joint.fit(initial, x, y, batches, weights, use_prefix)
        joint.guards(model, x[:9])
        with torch.no_grad():
            _, source = encoded(model, x)
            population = torch.stack([source[source_streams == s].mean(0) for s in np.unique(source_streams)]).mean(0)
            for person, r in held.items():
                assert not np.intersect1d(rows, r).size
                logits, items = encoded(model, torch.from_numpy(runtime._aligned(r, True)))
                for order in joint.ORDERS:
                    pos = streams[person][order][2]
                    assert np.array_equal(r[pos], perm[order][np.isin(perm[order], r)])
                    z, fe = logits[pos], items[pos]
                    own = model.predict(z, fe, use_prefix)
                    arms = {arm: own}
                    if use_prefix:
                        torch.testing.assert_close(own, external(model,z,fe,prior(fe)), rtol=0, atol=0)
                        dd = perm[order][np.isin(perm[order], donor)][:len(r)]
                        assert len(np.unique(dd)) == len(r)
                        fixed = population.expand(len(r), -1).clone()
                        fixed[0].zero_()
                        arms['population'] = external(model, z, fe, fixed)
                        arms['donor'] = external(model, z, fe, prior(source[np.searchsorted(rows, dd)]))
                        assert all(torch.equal(a[0], own[0]) for a in arms.values())
                    for name, pred in arms.items():
                        assert torch.isfinite(pred).all()
                        outputs[person,order,name] = pred.softmax(1).numpy()[np.argsort(pos)]
        print('complete arm', arm, flush=True)
    assert len(outputs) == 7*4*8 and traces['current']['initial_body'] == traces['prefix']['initial_body']
    np.savez_compressed('outputs.npz', **{'|'.join(k):v for k,v in outputs.items()})
    before = dict(source_sha=joint.sha(Path('source.sha256')), held_labels_opened=False, traces=traces,
                  output_sha=joint.sha(Path('outputs.npz')), training_rows=joint.digest(rows),
                  donor_rows=joint.digest(donor), source_sessions=len(np.unique(source_streams)), seed=374)
    Path('before_labels.json').write_text(json.dumps(before, indent=2))
    scorer = joint.official()
    joint.pin(Path(scorer.__file__), joint.METRIC_SHA)
    result = {}
    for ref in ('donor','population','A6','frozen','JS'):
        aligned = {**outputs, **{(p,o,'A6'):outputs[p,o,ref] for p in joint.COHORT for o in joint.ORDERS}}
        result[ref] = joint.score(scorer,joint.labels,meta,people,joint.COHORT,joint.ORDERS,aligned)
    result['seconds'] = time.monotonic()-start
    Path('result.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
