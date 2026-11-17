"""Pinned inputs and original §352 feature and §318 full-system guards."""
import csv
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path('/mnt/lustre/work/gehler/gfh527')
PROJECT = BASE / 'neural-interfaces-2026'
PAIR = BASE / 'ni26_pair352_20261117_v2'
PAIR_OUTPUT = PROJECT / 'runs/diag/external_pair352_v2'
CHAIN = BASE / 'ni26_s318_coral_fbl_20261117'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PAIR))
from data import CHANNELS, load_target, sha, source_orders

PAIR_PROOF_SHA = 'fc0cb9b8d9294cae9ae3b4c431c40c90259c6b444ec8343b19fbd304fcd9ee9f'
BASES_SHA = '331586168c265fe8051947052e69f08acb86e91182a6eefe2d19490724c827a4'
SCORER_SHA = 'ae387cd5a7b116fc1259001d4fe449ba1065b0cb5943c117d4f65ec87a0b40d4'
CONTROL_GUARD_SHA = 'f9d1f8cb1e33b74c51172fd859b39dea5c3cb2e5bd0e8c8a08d0eb6bdcd8eda6'
METRIC_PATH = BASE / 'ni26_s318_diagnostic_20261117/metrics.py'
METRIC_SHA = '25d25d32413a13dff6f728d29a46fe9b4c3f185c761595cb8341b570a4442d70'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path, expected):
    require(sha(path) == expected, f'changed pin: {path}')


def read_proof():
    pin(PAIR_OUTPUT / 'before_labels.json', PAIR_PROOF_SHA)
    proof = json.loads((PAIR_OUTPUT / 'before_labels.json').read_text())
    require(proof['target_labels_opened'] is False
            and proof['prefix_reset_interleaving_and_immutable_guards'], 'incomplete original §352 proof')
    for name, digest in proof['source_hashes'].items():
        pin(PAIR / name, digest)
    for name, digest in proof['inputs'].items():
        pin(PROJECT / name, digest)
    pin(PAIR_OUTPUT / 'source_predictions.npz', proof['outputs']['source_predictions.npz'])
    pin(CHAIN / 'data/proteus_windows.npz', proof['inputs']['data/proteus_windows.npz'])
    pin(CHAIN / 'runs/shuffle_perms.npz', proof['inputs']['runs/shuffle_perms.npz'])
    return proof


def proteus(proof):
    # Preparing the intervention never opens PROTEUS labels.
    with np.load(PROJECT / 'data/proteus_windows.npz', allow_pickle=False) as z:
        meta = {k: z[k] for k in ('subject', 'session', 'interface', 'task', 'run', 'ch_names')}
        channels = meta['ch_names'].tolist()
        x = z['X'][:, [channels.index(c) for c in CHANNELS]].copy()
    meta['person'] = np.array([p.rsplit('/', 1)[1] for p in meta['subject']])
    meta['stream'] = np.char.add(np.char.add(meta['subject'], '|'), meta['session'])
    meta['group'] = np.char.add(np.char.add(meta['task'], '|'), meta['run'].astype(str))
    orders = source_orders(PROJECT, meta)
    require(list(orders) == proof['orders'] and len(x) == proof['source_rows'] == 2655,
            'incomplete PROTEUS order or data inventory')
    exclusions = proof['source_exclusions']
    people = sorted(set(meta['subject']))
    require([r['person'] for r in exclusions] == people and len(people) == 10, 'LOSO people changed')
    for r in exclusions:
        rows = np.flatnonzero(meta['subject'] != r['person'])
        require(sorted(set(meta['subject'][rows])) == r['training_people']
                and hashlib.sha256(rows.tobytes()).hexdigest() == r['train_rows_sha256'],
                'original source LOPO mask changed')
    return x, meta, orders


def scherer():
    target = load_target(PROJECT)
    with np.load(PROJECT / 'data/scherer2015/windows.npz', allow_pickle=False) as z:
        labels = z['y']
    manifest = PROJECT / 'runs/diag/external_semantics351/scherer_rows.tsv.gz'
    with gzip.open(manifest, 'rt') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    require(np.array_equal(labels, [int(r['export_class']) for r in rows]), 'external labels changed')
    require(all(int(r['raw_class']) == {1: 2, 2: 1}[int(y)]
                for r, y in zip(rows, labels) if y in (1, 2)), 'external WORD/SUB labels reversed')
    require(np.bincount(labels).tolist() == [710, 710, 710]
            and len(set(target['subject'])) == 9 and len(set(target['stream'])) == 18,
            'external training inventory changed')
    return target, labels


def chain_guard(meta):
    score_dir = CHAIN / 'studies/14nov/newmember'
    guard_dir = CHAIN / 'studies/16nov/followups/coral_repair/chain/recovery'
    score_path = score_dir / 'score_s_v66p.py'
    pin(score_path, SCORER_SHA)
    pin(guard_dir / 'control_source.py', CONTROL_GUARD_SHA)
    sys.path[:0] = [str(score_dir), str(guard_dir)]
    from control_source import install
    spec = importlib.util.spec_from_file_location('score318', score_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    install(scorer, CHAIN)
    bases = CHAIN / 'runs/replay_data/bases.npz'
    pin(bases, BASES_SHA)
    with np.load(bases, allow_pickle=True) as z:
        person = z['person|all_loso']
        session = z['sess|all_loso']
        cell = z['cell|all_loso']
    require(np.array_equal(person, meta['person']), 'chain row owners differ')
    require(len(set(zip(session, meta['stream']))) == len(set(session)) == len(set(meta['stream'])),
            'chain session partition differs')
    context = np.char.add(np.char.add(meta['stream'], '|'), meta['interface'])
    require(len(set(zip(cell, context))) == len(set(cell)) == len(set(context)),
            'chain metric cells differ')
    return scorer, scorer.collect(CHAIN, len(person), len(set(session))), person, cell


def probability_rows(meta, scores, scorer):
    # v6.6 CP emits probabilities before its first positive fusion weight at i=90.
    pin(METRIC_PATH, METRIC_SHA)
    spec = importlib.util.spec_from_file_location('metrics318', METRIC_PATH)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    orders = source_orders(PROJECT, meta)
    stream = meta['stream']
    masks = {}
    for order, rows in orders.items():
        mask = np.zeros(len(rows), bool)
        for session in dict.fromkeys(stream):
            mask[rows[stream[rows] == session][:90]] = True
        for base in scorer.BASES:
            s = scores['control', order, base]
            require(np.array_equal((s >= 0).all(1), mask) and (s[~mask] <= 0).all(),
                    'actual score domain disagrees with the pinned CP session boundary')
            metrics.probability(s[mask])
        masks[order] = mask
    return masks
