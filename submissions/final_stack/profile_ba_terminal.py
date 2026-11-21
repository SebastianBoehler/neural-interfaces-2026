"""Pair MANAS eager and graph calls on the same incoming EEG windows."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

import check_hard_team_b as C
from check_ba_terminal import CANDIDATE, clean, unpack


def run(args):
    if os.environ['NI26_EXPECT_GPU'] not in torch.cuda.get_device_name():
        raise RuntimeError('unexpected Slurm GPU')
    C.verify_parent(args.data, C.DATA_SHA)
    data = C.D.load(args.data)
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, archive_sha=args.sha, sessions={})
    with TemporaryDirectory() as tmp:
        module, folder = unpack(args.zips, CANDIDATE, args.sha, Path(tmp))
        template = C.load(module, folder, data, 'cuda:0')
        warmup = data['X'][C.plans(data, '71000')['P02|1'][:32]]
        for k, (session, rows) in enumerate(C.D.plans(data, 71000, C.ZERO_STREAMS).items()):
            arms = ('eager', 'graph') if k % 2 == 0 else ('graph', 'eager')
            times = {}
            predictions = {}
            for arm in arms:
                model = C.fresh_model(template)
                if arm == 'eager':
                    model.team_a_graph = None
                for wave in warmup:
                    model._team_a_probability(wave)
                model.reset_state()
                if arm == 'eager':
                    model.team_a_graph = None
                elapsed, posteriors = [], []
                for row in rows:
                    torch.cuda.synchronize()
                    then = perf_counter()
                    p = model._team_a_probability(data['X'][row])
                    torch.cuda.synchronize()
                    elapsed.append(1000 * (perf_counter() - then))
                    posteriors.append(p)
                if model.team_a.age != len(rows):
                    raise RuntimeError('MANAS skipped a window')
                if arm == 'graph' and model.team_a_graph.replays != len(rows):
                    raise RuntimeError('MANAS graph was inactive')
                times[arm] = dict(mean_ms=float(np.mean(elapsed)), warm_ms=float(np.mean(elapsed[1:])),
                                  capture_window_ms=elapsed[0])
                predictions[arm] = np.stack(posteriors)
            error = float(np.max(np.abs(predictions['eager'] - predictions['graph'])))
            if error > 2e-5 or not np.array_equal(predictions['eager'].argmax(1), predictions['graph'].argmax(1)):
                raise RuntimeError(f'{session}: MANAS graph differs from eager, max diff {error}')
            report['sessions'][session] = dict(windows=len(rows), max_abs=error, **times)
        clean(template)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('zips', type=Path)
    parser.add_argument('data', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--sha', required=True)
    run(parser.parse_args())
