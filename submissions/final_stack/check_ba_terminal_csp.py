"""Verify the hybrid on a pinned Scherer stream with active CSP score fusion."""
import argparse
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

import check_hard_team_b as C
from check_ba_terminal import clean, play, unpack

SCHERER_SHA = 'a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91'


def main(run, sha, diagnose, eager_first):
    source = run / 'src/data/scherer2015/windows.npz'
    C.verify_parent(source, SCHERER_SHA)
    tape = run / 'niv/out/ba/smoke.npz'
    with np.load(tape, allow_pickle=False) as prior, np.load(source, allow_pickle=False) as data:
        key = 'full|0|Scherer2015/G|2'
        rows, labels = prior[key + '|rows'], prior[key + '|labels']
        if (len(rows) != 120 or len(np.unique(rows)) != 120
                or not np.all(data['subject'][rows] == 'Scherer2015/G') or not np.all(data['session'][rows] == '2')):
            raise ValueError('pinned Scherer stream differs')
        metadata = dict(ch_names=list(data['ch_names']))
        x = torch.from_numpy(data['X'][rows]).to('cuda:0')
    with TemporaryDirectory() as scratch:
        module, folder = unpack(run / 'zips', 'ba_terminal_niv.zip', sha, Path(scratch))
        model = C.load(module, folder, metadata, 'cuda:0')
        arms = [('graph', True), ('eager', False)]
        if diagnose:
            arms += [('eager_repeat', False), ('graph_repeat', True)]
        if eager_first:
            arms = [arms[i] for i in (1, 0, 3, 2)]
        results = {}
        posterior = type(model)._team_a_probability
        for name, use_graph in arms:
            captured = []

            def record(session, raw):
                q = posterior(session, raw)
                captured.append(q.copy())
                return q

            with patch.object(type(model), '_team_a_probability', record):
                got, scores, meta = play(model, x, capture=True, use_graph=use_graph)
            np.testing.assert_array_equal(got, labels)
            if meta['fused'] <= 0 or len(captured) != len(rows):
                raise RuntimeError(f'CSP or Team A path was inactive: {meta}')
            results[name] = (scores, np.stack(captured), meta)
        clean(model)
    spans = dict(mix=slice(0, 3), route=slice(3, 6), terminal=slice(6, 9))
    comparisons = {}
    for a, b in combinations(results, 2):
        scores, manas, meta = results[a]
        reference, ref_manas, ref_meta = results[b]
        if meta != ref_meta:
            raise RuntimeError(f'stage counts differ: {meta}, {ref_meta}')
        if not diagnose:
            np.testing.assert_allclose(scores, reference, rtol=0, atol=2e-5)
        np.testing.assert_allclose(manas, ref_manas, rtol=0, atol=2e-5)
        differences = np.abs(scores - reference)
        comparisons[f'{a}/{b}'] = {name: float(differences[:, span].max()) for name, span in spans.items()}
        comparisons[f'{a}/{b}']['team_a'] = float(np.abs(manas - ref_manas).max())
    print(json.dumps(dict(windows=len(rows), pinned_labels_equal=True, csp_fused=meta['fused'],
                          maximum_absolute_difference=comparisons)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('sha')
    p.add_argument('--diagnose', action='store_true')
    p.add_argument('--eager-first', action='store_true')
    a = p.parse_args()
    if a.eager_first and not a.diagnose:
        p.error('--eager-first requires --diagnose')
    main(a.run, a.sha, a.diagnose, a.eager_first)
