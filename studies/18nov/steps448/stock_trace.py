"""Distinguish stock-only reset sensitivity from mutations of serialized tensors."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import replay
from benchmark_utils.streaming import fresh_model


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(root):
    fields = {}
    visited = set()

    def visit(obj, name):
        if isinstance(obj, torch.Tensor):
            fields[name] = digest(obj.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(obj, np.ndarray):
            fields[name] = digest(obj.tobytes())
        elif isinstance(obj, (int, float, bool, str, np.generic)):
            fields[name] = str(obj)
        elif id(obj) not in visited:
            visited.add(id(obj))
            if isinstance(obj, torch.nn.Module):
                for k, v in obj.state_dict().items():
                    visit(v, f'{name}.{k}')
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    visit(v, f'{name}.{i}')
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    visit(v, f'{name}.{k}')
            elif type(obj).__module__ == 'runtime_solver':
                visit(vars(obj), name)
    visit(root, 'model')
    assert len(fields) > 500
    return fields


def observe(template, x, rows):
    model = fresh_model(template)
    log = []
    step = model.cp.step

    def record(*args):
        out = step(*args)
        log.append(dict(out=out.tolist(), pre=args[1].tolist(), filter=args[2].tolist(),
                        mix=model.trace['mix'].tolist(),
                        members=[v.tolist() for _, v in sorted(model.trace['v6']['members'].items())],
                        fine=[v.tolist() for _, v in sorted(model.trace['v6']['p_ft'].items())]))
        return out

    model.cp.step = record
    labels = [int(model.predict(x[r:r+1].to('cuda:0'))[0]) for r in rows]
    assert labels == [int(np.argmax(row['out'])) for row in log]
    assert replay.activity(model)['errors'] == 0
    return labels, log


def compare(ref, records, reference):
    labels, baseline = ref
    pred, rows = records
    wrong = np.flatnonzero(np.array(labels) != pred)
    first = {}
    for field in ('out', 'pre', 'filter', 'mix', 'members', 'fine'):
        for i, (a, b) in enumerate(zip(baseline, rows)):
            x, y = np.asarray(a[field]), np.asarray(b[field])
            if not np.array_equal(x, y):
                first[field] = [i, float(np.abs(x-y).max())]
                break
    return dict(canonical_mismatches=np.flatnonzero(np.array(pred) != reference).tolist(),
                first_field_delta=first, switches=[
                    dict(index=int(i), cold=baseline[i]['out'], observed=rows[i]['out']) for i in wrong])


def main():
    prior = json.loads(Path('/prior/cap_5.json').read_text())
    first, second, *_ = list(prior.values())
    rows = np.array(first['rows'])
    expected = first['labels']
    with np.load('/external.npz') as archive:
        x = torch.from_numpy(archive['X'])
        names = archive['ch_names'].tolist()
    template = replay.load(Path('/package'), names)
    before = snapshot(template)
    baseline = observe(template, x, rows)
    assert baseline[0] == expected
    result = {}
    for name, template_here in [('same_second_copy', template), ('different_prior_day', template),
                                ('fresh_loaded_template', None)]:
        if name == 'different_prior_day':
            observe(template, x, np.array(second['rows']))
        if template_here is None:
            template_here = replay.load(Path('/package'), names)
        rec = observe(template_here, x, rows)
        after = snapshot(template)
        changed = [k for k in sorted(before.keys() | after.keys()) if before.get(k) != after.get(k)]
        result[name] = dict(**compare(baseline, rec, expected), changed_template_fields=changed,
                            checked_fields=len(before), rows=rows.tolist())
        Path('stock_trace.json').write_text(json.dumps(result))


if __name__ == '__main__':
    main()
