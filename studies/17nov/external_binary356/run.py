"""One fixed external expert, then guarded scoring of the real current chain."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

from decision import decisions, odds
from inputs import (CHAIN, PAIR_OUTPUT, PROJECT, chain_guard, pin, probability_rows, proteus,
                    read_proof, require, scherer, sha)
from geometry import Geometry, digest, fit
from producer import banks

FEATURES_SHA = '360f335d5a9224aad67af9725907cda07ce799fb574a62ae44bb7eb6cbe3dabd'
DOSES = dict(zero=(0., 0.), source=(.15, 0.), external=(.15, .10), reverse=(.15, -.10))


def build_expert(out, proof, smoke):
    x, meta, orders = proteus(proof)
    people = sorted(set(meta['person']))
    target, y = scherer()
    geometry = Geometry(np.concatenate([x, target['X']]), PROJECT)
    stream = np.r_[meta['stream'], target['stream']]
    external = np.arange(len(x), len(stream))
    features = geometry.training(external, stream)['cov']
    model = fit({'cov': features}, y)
    checksum = digest(model)
    require(checksum == digest(fit({'cov': features}, y)), 'frozen external model refit changed')
    predictions = {name: np.full((len(x), 2), np.nan) for name in orders}
    elapsed = []
    basis = {}
    for person in people[:1] if smoke else people:
        start = time.monotonic()
        query = np.flatnonzero(meta['person'] == person)
        prior = np.flatnonzero(meta['person'] != person)
        require(np.intersect1d(query, prior).size == 0 and len(set(meta['person'][prior])) == 9,
                'held query leaked into alignment prior')
        prior_hash = hashlib.sha256(prior.tobytes()).hexdigest()
        expected = next(r for r in proof['source_exclusions'] if r['person'].endswith('/' + person))
        require(prior_hash == expected['train_rows_sha256'], 'alignment source exclusion changed')
        basis[person] = prior_hash
        values = banks(geometry, query, meta['stream'], orders, geometry.prior(prior), model)
        for name in orders:
            predictions[name][query] = values[f'cov|{name}'][query]
        elapsed.append(time.monotonic() - start)
        print(f'{person} external one-window banks pass in {elapsed[-1]:.2f}s', flush=True)
    require(digest(model) == checksum, 'external model changed at inference')
    select = meta['person'] == people[0] if smoke else np.ones(len(x), bool)
    with np.load(PAIR_OUTPUT / 'source_predictions.npz', allow_pickle=False) as z:
        in_domain = {order: odds(z['cov|' + order][select]) for order in orders}
    external_log_odds = {order: odds(p[select]) for order, p in predictions.items()}
    for name in orders:
        require(np.ptp(external_log_odds[name]) > .1, 'inactive external logits')
    np.savez_compressed(out / 'expert_odds.npz', **{f'{name}|{arm}': x for arm, bank in
                        [('source', in_domain), ('external', external_log_odds)] for name, x in bank.items()})
    return meta, select, in_domain, external_log_odds, basis, checksum, elapsed


def consume(meta, select, in_domain, external):
    scorer, scores, people, cells = chain_guard(meta)
    probabilities = probability_rows(meta, scores, scorer)
    predictions, changes = {}, {}
    for order in scorer.ORDERS:
        for base in scorer.BASES:
            s = scores['control', order, base][select]
            reference = s.argmax(1)
            by_arm = {arm: decisions(s, probabilities[order][select], in_domain[order], external[order], *dose)
                      for arm, dose in DOSES.items()}
            require(np.array_equal(reference, by_arm['zero']), 'zero differs from current chain')
            require(all(np.array_equal(p == 0, reference == 0) for p in by_arm.values()), 'MI decisions changed')
            for arm, p in by_arm.items():
                predictions[order, base, arm] = p
            changes[order, base] = {a: int(np.sum(by_arm[a] != by_arm[b])) for a, b in
                                    [('source', 'zero'), ('external', 'source'), ('reverse', 'source')]}
    return scorer, predictions, changes, people[select], cells[select]


def score(scorer, predictions, people, cells):
    with np.load(CHAIN / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        y = z['y|all_loso']
    with np.load(PROJECT / 'data/proteus_windows.npz', allow_pickle=False) as z:
        require(np.array_equal(y, z['y']), 'chain/source row labels differ')
    cell_counts = np.array([len(np.unique(cells[people == p])) for p in scorer.PEOPLE])
    summary = {}
    for bases in (scorer.BASES, ('B2_pool',)):
        values = {arm: np.stack([np.mean([scorer.person_accuracy(
                    np.eye(3)[predictions[order, base, arm]], y, cells, people, scorer.PEOPLE)
                    for base in bases], axis=0) for order in scorer.ORDERS]) for arm in DOSES}
        summarize = lambda v: dict(people_order=scorer.summarize(v, scorer.ORDERS, scorer.PEOPLE),
                                   official_cell=scorer.cell_weighted_summary(v, scorer.ORDERS, cell_counts))
        by_arm = {a: summarize(v) for a, v in values.items()}
        frozen_baseline = .76217673 if len(bases) == 6 else .762793
        require(abs(by_arm['zero']['official_cell']['mean'] / 100 - frozen_baseline) < 1e-6,
                'unchanged §318 control metric changed')
        contrasts = {f'{a}_minus_{b}': summarize(values[a] - values[b]) for a, b in
                     [('source', 'zero'), ('external', 'zero'), ('external', 'source'), ('external', 'reverse')]}
        summary['six_base_mean' if len(bases) == 6 else bases[0]] = dict(arms=by_arm, contrasts=contrasts)
    return summary


def write_json(path, payload):
    with path.open('x') as f:
        f.write(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    started = time.monotonic()
    args.out.mkdir(exist_ok=False)
    pin(PROJECT / 'experiments/features.py', FEATURES_SHA)
    proof = read_proof()
    meta, select, source, external, alignment, weight_sha, elapsed = build_expert(args.out, proof, args.smoke)
    scorer, predictions, changes, people, cells = consume(meta, select, source, external)
    totals = {a: sum(v[a] for v in changes.values()) for a in ('source', 'external', 'reverse')}
    require(all(v > 0 for v in totals.values()), 'terminal arm has zero actual decision changes')
    pin(PROJECT / 'experiments/features.py', FEATURES_SHA)
    full_wall = time.monotonic() - started
    projected_wall = full_wall + 14 * max(elapsed) + 90
    write_json(args.out / 'before_labels.json', dict(
        scope='smoke P01 only' if args.smoke else 'complete 10 LOSO people and 8 pinned causal orders',
        proteus_labels_opened=False, trained_on='Scherer only, 710 WORD and 710 SUB rows',
        frozen_external_model_sha256=weight_sha, alignment_prior_rows_sha256=alignment,
        profile_fold_wall_seconds=elapsed, wall_seconds=full_wall,
        conservative_full_projection_seconds=projected_wall if args.smoke else None,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, decision_changes=totals,
        individual_bank_sha256=sha(args.out / 'expert_odds.npz'),
        chains=48, zero_identity=True, mi_decisions_exact=True, cp_score_domains_exact=True,
        permission='11-Nov organizer clearance'))
    print(json.dumps(totals), flush=True)
    if not args.smoke:
        write_json(args.out / 'result.json', score(scorer, predictions, people, cells))


if __name__ == '__main__':
    main()
