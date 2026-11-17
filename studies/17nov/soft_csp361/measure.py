"""Score the complete guarded chain with the unchanged official metric."""
import importlib.util

import numpy as np

from assets import CHAIN, ROOT, pin, require

METRIC = ROOT/'ni26_raw312_cpu_20261117_v2/experiments/protocol.py'
METRIC_SHA = '390fd6bea0ddc9c53e34fdabb094b1a269ff45b07b4a5542a8135b44d5314a2c'


def official():
    pin(METRIC, METRIC_SHA)
    spec = importlib.util.spec_from_file_location('official_protocol361', METRIC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.score


def summarize(scorer, scores, metadata, people):
    metric = official()
    with np.load(CHAIN/'data/proteus_windows.npz', allow_pickle=False) as z:
        labels = z['y']
    bases = (scorer.BASES, ('B2_pool',))
    report = {}
    for selected in bases:
        metrics = np.zeros((2, len(scorer.ORDERS), len(scorer.PEOPLE)))
        counts = np.zeros(len(scorer.PEOPLE), dtype=np.int64)
        for oi, order in enumerate(scorer.ORDERS):
            for pi, person in enumerate(scorer.PEOPLE):
                mask = people == person
                for a, arm in enumerate(('control', 'fractional')):
                    for base in selected:
                        score = metric(labels[mask], scores[arm, order, base][mask].argmax(1), metadata, mask)
                        counts[pi] = score['n_cells']
                        metrics[a, oi, pi] += 100 * score['balanced_accuracy_cell_mean'] / len(selected)
        summary = {}
        difference = metrics[1] - metrics[0]
        for name, values in [('control', metrics[0]), ('fractional', metrics[1]), ('fractional_minus_control', difference)]:
            summary[name] = dict(official=scorer.cell_weighted_summary(values, scorer.ORDERS, counts),
                                 people_order=scorer.summarize(values, scorer.ORDERS, scorer.PEOPLE))
        anchor = 76.21767294925993 if len(selected) == 6 else 76.27933502145758
        require(abs(summary['control']['official']['mean']-anchor) < 1e-7, 'official control metric changed')
        report['all_bases' if len(selected) == 6 else 'B2_pool'] = summary
    return report
