"""Real source-person guards: strong raw recipe, paired control and active new supervision."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ['NI26_WS'])
sys.path.insert(0, str(ROOT / 'experiments'))
import protocol
import session_bn
from data import sha
from training import fit, probabilities, sources


def main():
    device = os.environ['NI26_DEVICE']
    if device not in ('cpu', 'cuda') or device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('explicit device unavailable')
    torch.set_num_threads(6)
    d = protocol.load(); target, available, val = sources(d, 'P01', 0)
    people = [p for p in available if p != val]
    rows = np.flatnonzero(np.isin(d['subject'], people))
    test = d['subject'] == val
    # Existing strong raw SessionBN recipe, unchanged builder and training loop.
    exact = session_bn.make('fbcnet_raw_sbn', d, 0)
    exact.fit(rows); exact.set_k(20.)
    if exact.fit_subjects != set(people) or target in exact.fit_subjects:
        raise ValueError('strong recipe source exclusion failed')
    with torch.no_grad():
        reference = protocol.predict_streams(exact, d, test, proba=True)
        r = np.flatnonzero(test & (d['stream'] == d['stream'][np.flatnonzero(test)[0]]))
        streamed = exact.predict_proba_stream(r)
        direct = torch.softmax(exact.net(torch.from_numpy(exact._aligned(r, True)).to(device))
                               [:, :3].cpu().double(), 1).numpy()
    if not np.array_equal(streamed, direct):
        raise ValueError('exact raw SessionNet forward/harness path mismatch')
    def ba(p):
        labels = d['y'][test]
        return float(np.mean([(p.argmax(1)[labels == c] == c).mean() for c in (0, 1, 2)]))
    report = dict(device=device, held_excluded=target, validation_person=val, fit_people=people,
                  exact_recipe='fbcnet_raw_sbn;k20;epochs40;unchanged SessionNet',
                  exact_source_validation_ba=ba(reference), models={}, checks=[])
    del exact
    cache = Path(os.environ['NI26_CACHE'])
    for arch in ('fb', 'sh'):
        net, scale, meta = fit(d, cache, people, arch, 'S', 0, device, val_people=[val])
        pred = probabilities(net, d['X'][test], scale, device)
        report['models'][arch] = dict(source_validation_ba=ba(pred), metadata=meta)
        if arch == 'fb' and ba(pred) < report['exact_source_validation_ba'] - .05:
            raise ValueError(f'matched strong control loses >5pp on source validation: {report}')
        del net
        for arm in ('F', 'T', 'C'):
            net, scale, meta = fit(d, cache, people, arch, arm, 0, device,
                                  epochs=1, val_people=[val], pilot=True)
            p = probabilities(net, d['X'][test][:17], scale, device)
            if p.shape != (17, 3):
                raise ValueError('source prediction smoke failed')
            report['checks'].append(dict(architecture=arch, arm=arm, metadata=meta))
            del net
    report['source_manifest'] = sha(ROOT / 'source.sha256')
    report['passed'] = True
    output = ROOT / f'guard_{device}.json'
    with output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
