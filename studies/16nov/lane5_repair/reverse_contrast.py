"""Post-hoc paired intervals from the complete validated §303 matrix."""
import importlib.util
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
ANALYSIS = Path(os.environ['ANALYSIS_ROOT']).resolve(strict=True)
spec = importlib.util.spec_from_file_location('repair_score', ROOT / 'studies/16nov/lane5_repair/score.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def main():
    manifest = ROOT / 'repair.sha256'; library = ROOT / 'runs/build/liblane5.so'
    extension = ROOT / 'runs/build/liblane5_repair.so'
    result = ROOT / 'evidence/next_wave_16nov/lane5_repair_results.json'
    if (s.sha(manifest) != '060625ce4de3a8a9e2a88313303a507a2451f4bf1d9cd13c052fe2601fa1b856' or
            s.sha(result) != 'd836609fa94f0c3a32aecd107550b685f6ada46eb32909f06630a15fb5630740'):
        raise ValueError('original result or manifest mismatch')
    s.old.check_manifest(ROOT, ROOT / 'source.sha256', library)
    d = s.old.without_labels(ROOT)
    original, orig_hashes = s.old.load_outputs(d, ROOT / 'source.sha256', library)
    values, hashes, _, _ = s.load_complete(d, manifest, library, extension, original)
    previous = json.loads(result.read_text())
    if hashes != previous['output_sha256'] or orig_hashes != previous['original_output_sha256']:
        raise ValueError('scored output matrix changed')
    # Read labels only after both complete matrices and their hashes passed.
    with np.load(ROOT / 'data/proteus_windows.npz', allow_pickle=False) as z: y = z['y'].copy()
    person = np.array([x.split('/')[-1] for x in d['subject']])
    cell = np.char.add(np.char.add(d['stream'], '|'), d['interface'])
    helper = s.old.helpers(); contrasts = {}
    for arm, _, _ in s.ARMS:
        pp = {}
        for order in s.ORDERS:
            for mode in ('static', 'adaptive', 'reverse'):
                pp[order, mode] = np.mean([helper.person_accuracy(values[arm, seed, order, mode], y, cell, person, s.PEOPLE)
                                           for seed in range(2)], axis=0)
        for right in ('static', 'adaptive'):
            delta = np.stack([pp[o, 'reverse'] - pp[o, right] for o in s.ORDERS])
            contrasts[f'{arm}_reverse-minus-{right}'] = helper.summarize(delta, s.ORDERS, s.PEOPLE)
    output = ANALYSIS / 'results.json'
    if output.exists(): raise FileExistsError(output)
    with output.open('x') as f:
        json.dump(dict(contrast_type='post-hoc exploratory; points previously observed',
                       original_result_sha256=s.sha(result), contrasts=contrasts), f, indent=2, sort_keys=True)
        f.write('\n')
    print('validated exploratory paired contrasts:', output, 'sha256', s.sha(output), flush=True)


if __name__ == '__main__': main()
