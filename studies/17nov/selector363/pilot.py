"""One nested held-person classifier over existing full-system disagreements."""
import json
import sys
import time
from pathlib import Path

import numpy as np

OWNER = Path('/mnt/lustre/work/gehler/gfh527/ni26_soft_csp361_v1_20261117')
sys.path.insert(0, str(OWNER))
from assets import CHAIN, ROOT, chain_guard, pin, require, sha, verify_sources
from measure import official
from inputs import probability_rows
from selector import fit, four_margins, margin_threshold, predict

LABEL_READER = ROOT/'ni26_s306_expert_cpu_v2_20261116/source_labels.py'
CEILING = ROOT/'ni26_ceiling362_v1_20261117/result.json'


def data():
    verify_sources()
    pin(OWNER/'source.sha256', '9e889431c92f4759c79e0724446d19fc9503ac5b726464f73f9ed22b548f9ccd')
    pin(CEILING, '5d31eefa52e7239a8fe942b80a931e75613c0227e5959b0c4822e681c1dbdd26')
    with np.load(CHAIN/'data/proteus_windows.npz', allow_pickle=False) as z:
        meta = {k: z[k] for k in ('subject', 'session', 'interface', 'task', 'run')}
    meta['person'] = np.array([p.rsplit('/', 1)[-1] for p in meta['subject']])
    meta['stream'] = np.char.add(np.char.add(meta['subject'], '|'), meta['session'])
    meta['group'] = np.char.add(np.char.add(meta['task'], '|'), meta['run'].astype(str))
    meta['ch_names'] = None
    scorer, scores, people, cells = chain_guard(meta)
    modes = probability_rows(meta, scores, scorer)
    arms = np.stack([[[scores[a, o, b] for b in scorer.BASES] for o in scorer.ORDERS]
                     for a in ('control', 'JS')])
    domain = np.stack([modes[o] for o in scorer.ORDERS])[:, None, :]
    require(np.array_equal((arms >= 0).all(-1), np.broadcast_to(domain, arms.shape[:-1])),
            'complete branches have inconsistent causal score domains')
    shape = arms.shape[1:-1]
    encoded = np.where(domain[..., None], np.log(np.clip(arms, 1e-6, 1)), arms)
    a, b = arms.argmax(-1)
    disagreement = a != b
    x = four_margins(encoded[0][disagreement], encoded[1][disagreement], a[disagreement], b[disagreement])
    rows = np.broadcast_to(np.arange(len(people)), shape)[disagreement]
    scheme_weights = np.asarray([.25]*4 + [.5]*4)
    row_weights = np.broadcast_to(scheme_weights[:, None, None], shape)[disagreement]
    require(np.isfinite(x).all() and len(x) == 4021 and len(set(rows)) == 987, 'complete pair/disagreement identity')
    return meta, people, cells, scorer, a, b, disagreement, rows, row_weights, x


def read_labels(rows, size):
    import importlib.util
    pin(LABEL_READER, '35edab56fe754e6c77a81f54f2010baca7e0912af55af9564043b2a8bde8c4dc')
    spec = importlib.util.spec_from_file_location('source_labels363', LABEL_READER)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    return reader.read_training_labels(CHAIN/'data/proteus_windows.npz', rows, size)


def fold(held, people, cells, rows, order_weights, x, base, extra):
    available = people != held
    source = np.flatnonzero(available)
    labels = np.full(len(people), -1, dtype=int)
    labels[source] = read_labels(source, len(people))
    require((labels[~available] == -1).all(), 'held labels entered readout fitting')
    row_score = np.zeros(len(people))
    for cell in np.unique(cells[source]):
        for value in (0, 1, 2):
            idx = np.flatnonzero(available & (cells == cell) & (labels == value))
            require(len(idx) > 0, 'missing source class changes official cost')
            row_score[idx] = 1 / len(idx)
    owner = people[rows]
    yes, no = extra == labels[rows], base == labels[rows]
    decisive = (yes | no) & available[rows]
    fit_weight = order_weights * row_score[rows]
    out_of_fold = np.zeros(len(rows), dtype=bool)
    observed = np.zeros(len(rows), dtype=bool)
    for excluded in sorted(set(people[source])):
        train = decisive & (owner != excluded)
        validation = owner == excluded
        model = fit(x[train], yes[train], fit_weight[train])
        out_of_fold[validation] = predict(model, x[validation])
        observed[validation] = True
    require(np.array_equal(observed, available[rows]), 'source OOF split incomplete')
    margin = x[:, 1] - x[:, 0]
    cut, rate = margin_threshold(margin[observed], order_weights[observed], out_of_fold[observed])
    model = fit(x[decisive], yes[decisive], fit_weight[decisive])
    held_disagreement = owner == held
    selected = predict(model, x[held_disagreement])
    fixed = margin[held_disagreement] >= cut
    return held_disagreement, selected, fixed, dict(
        fit_people=sorted(set(people[source])), source_decisive=int(decisive.sum()),
        held_disagreements=int(held_disagreement.sum()), selector_JS=int(selected.sum()),
        fixed_JS=int(fixed.sum()), source_OOF_JS_rate=rate, margin_cut=cut if np.isfinite(cut) else None,
        coefficients=model[1].coef_[0].tolist(), source_only=True)


def score(meta, people, scorer, pred):
    y = read_labels(np.arange(len(people)), len(people))
    metric = official()
    summaries = {}
    for name, bases in [('six_bases', range(6)), ('B2_pool', range(1))]:
        data = np.empty((4, 8, len(scorer.PEOPLE)))
        counts = np.zeros(len(scorer.PEOPLE), dtype=int)
        for i, person in enumerate(scorer.PEOPLE):
            keep = people == person
            for a in range(4):
                for o in range(8):
                    rows = [metric(y[keep], pred[a][o, b][keep], meta, keep) for b in bases]
                    data[a, o, i] = 100 * np.mean([v['balanced_accuracy_cell_mean'] for v in rows])
                    counts[i] = rows[0]['n_cells']
        record = {a: scorer.cell_weighted_summary(data[i], scorer.ORDERS, counts)
                  for i, a in enumerate(('base', 'JS', 'fixed', 'selector'))}
        for i, a in enumerate(('base', 'JS', 'fixed')):
            contrast = data[3]-data[i]
            record['selector_minus_'+a] = dict(
                official=scorer.cell_weighted_summary(contrast, scorer.ORDERS, counts),
                people_order=scorer.summarize(contrast, scorer.ORDERS, scorer.PEOPLE))
        summaries[name] = record
    require(abs(summaries['six_bases']['base']['mean']-76.21767294925993)<1e-8
            and abs(summaries['B2_pool']['JS']['mean']-76.55173151211109)<1e-8, 'frozen endpoint mismatch')
    return summaries


def emit(a, b, disagreement, choose):
    picked = a.copy()
    picked[disagreement] = np.where(choose, b[disagreement], a[disagreement])
    return picked


def main():
    start = time.monotonic()
    meta, people, cells, scorer, a, b, disagree, rows, weights, x = data()
    learned, fixed = np.zeros(len(x), bool), np.zeros(len(x), bool)
    covered = np.zeros(len(x), bool)
    folds = {}
    for held in scorer.PEOPLE:
        pos, ml, mf, folds[held] = fold(held, people, cells, rows, weights, x, a[disagree], b[disagree])
        require(not covered[pos].any(), 'held fold routing collision')
        covered[pos] = True
        learned[pos], fixed[pos] = ml, mf
        print(f'{held}: all orders, source nested gate complete', flush=True)
    require(covered.all() and learned.any() and (~learned).any(), 'gate incomplete or inactive')
    p, q = emit(a, b, disagree, fixed), emit(a, b, disagree, learned)
    require(np.array_equal(emit(a, b, disagree, np.zeros_like(learned)), a)
            and np.array_equal(emit(a, b, disagree, np.ones_like(learned)), b), 'terminal endpoints changed')
    require(np.all((q == a) | (q == b)) and np.array_equal(q[~disagree], a[~disagree]),
            'noncausal/unsupported terminal branch')
    np.savez_compressed('predictions.npz', base=a, JS=b, fixed=p, selector=q)
    proof = dict(folds=folds, outputs_sha256=sha(Path('predictions.npz')), held_scoring_labels_opened=False,
                 source_sha256=sha(Path('source.sha256')), selector_JS_total=int(learned.sum()),
                 fixed_JS_total=int(fixed.sum()), disagreement_total=len(x),
                 selector_JS_B2_pool=int(np.sum(disagree[:, 0] & (q[:, 0] == b[:, 0]))),
                 fixed_JS_B2_pool=int(np.sum(disagree[:, 0] & (p[:, 0] == b[:, 0]))), causal_current_only=True)
    with Path('before_labels.json').open('x') as f:
        json.dump(proof, f, indent=2)
    result = score(meta, people, scorer, (a, b, p, q))
    result.update(proof_sha256=sha(Path('before_labels.json')), task_wall_s=time.monotonic()-start)
    with Path('result.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result['B2_pool']['selector_minus_JS']), flush=True)


if __name__ == '__main__':
    main()
