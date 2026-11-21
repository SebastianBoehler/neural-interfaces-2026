"""Audit the exact Team A fold weights, fit populations and frozen statistics."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

PUBLIC_SHA = '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
INIT_SHA = '5c7fb0a1de75bf01c179fdc416a4e4af3ad4ead7895ac03b7e357adb587cfa64'
BATCH = 24


def read_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_artifact(directory, *, public=None, transfer=None, external=None):
    from data import sha

    m = read_json(directory / 'training.json')
    path = directory / 'weights.pt'
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    identity = ('arch', 'recipe', 'seed', 'smoke', 'inputs')
    require(all(m[k] == payload[k] for k in identity) and tuple(m['corpora']) == payload['corpora'], directory)
    require((m['arch'], m['recipe'], m['seed'], m['smoke'], m['init_sha']) == ('manas', 'A1', 0, False, INIT_SHA), directory)
    require(sha(path) == m['weights_sha'] and m['public_sha'] == public and m['transfer_sha'] == transfer, directory)
    if external is not None:
        require((m['corpora'], m['inputs']) == (external['corpora'], external['inputs']), directory)
    return m, payload


def external_fit(directory, public_file, extra, log, scherer_free):
    """Repeat the external loader's row selection and sample the actual update draws."""
    from data import sha
    from fit import sample_weights
    from types import SimpleNamespace

    m, payload = verify_artifact(directory)
    require((m['people'], m['first_session_only'], m['steps'], m['epochs']) == ([], [], 1152, 0), directory)
    require('bank' not in payload and ('scherer2015' in m['corpora']) != scherer_free, directory)
    rng = np.random.default_rng(m['seed'])
    data = {}
    pins = {}
    for corpus in m['corpora']:
        for root in (extra, public_file.parent):
            folder = root / corpus
            files = sorted(folder.glob('windows_*.npz'))
            if not files and (folder / 'windows.npz').is_file():
                files = [folder / 'windows.npz']
            if files:
                break
        require(bool(files), corpus)
        fields = []
        for path in files:
            key = f'{corpus}/{path.name}'
            pins[key] = sha(path)
            require(m['inputs'][key] == pins[key], key)
            with np.load(path, allow_pickle=False) as src:
                people, visit, label = src['subject'].astype(str), src['session'].astype(str), src['y']
                keep = src['win'] == 0 if 'win' in src else np.ones(len(people), dtype=bool)
                require(not any(p.startswith('Dreyer2026Proteus/') for p in people[keep]), f'PROTEUS in {key}')
                groups = []
                for person in np.unique(people[keep]):
                    who = keep & (people == person)
                    for cls in np.unique(label[who]):
                        rows = np.flatnonzero(who & (label == cls))
                        groups.extend(rng.choice(rows, min(len(rows), 64), replace=False))
                if groups:
                    rows = np.sort(groups)
                    fields.append((people[rows], visit[rows], label[rows]))
        p, v, y = map(np.concatenate, zip(*fields, strict=True))
        require(re.search(rf'(?m)^{corpus} {len(y)} ', log.read_text()), f'{corpus}: fit log row count')
        data[corpus] = SimpleNamespace(person=p, visit=v, y=y, interface=np.repeat(corpus, len(y)))
    require(pins == m['inputs'], 'external manifest covers different shards')
    order = tuple(data)
    probabilities = {name: sample_weights(d) for name, d in data.items()}
    actual = {name: Counter() for name in order}
    for step in range(m['steps']):
        name = order[step % len(order)]
        d = data[name]
        rows = rng.choice(len(d.y), BATCH, p=probabilities[name])
        actual[name].update(f'{d.person[i]}|{d.visit[i]}' for i in rows)
        advance_augmentation(rng, min(12, len(set(zip(d.person, d.visit)))) ** 2)
    return dict(training_sha=sha(directory / 'training.json'), weights_sha=m['weights_sha'], shards=pins,
                corpora={name: dict(sampled=int(len(d.y)), people=len({p for p in d.person}),
                                   windows_used_by_session=dict(sorted(actual[name].items()))) for name, d in data.items()})


def advance_augmentation(rng, mixes):
    if rng.random() < 0.5:
        rng.integers(mixes)
        rng.uniform(0.03, 0.09)
    rng.integers(-22, 23)


def audit_fold(fold, public_file, run, lane, external):
    from data import PEOPLE, read_public, sha, source_rows
    from fit import sample_weights
    from gallery import bank
    from package import load_source

    if fold == 'production':
        people, first = list(PEOPLE), []
    else:
        mode, held = fold.split('_')
        require(mode == 'new' or (mode == 'ret' and held in ('P04', 'P10')), fold)
        people = [p for p in PEOPLE if p != held or mode == 'ret']
        first = [held] if mode == 'ret' else []
        jobs = [line.split('\t') for line in (run / 'jobs.tsv').read_text().splitlines()]
        job = next(j for j in jobs if j[:2] == [mode, held])
        require(job[2].split(',') == people and job[3] == ','.join(first), 'Slurm fold row differs')
    data = read_public(public_file)
    check = np.flatnonzero(np.isin(data.person, people) & (~np.isin(data.person, first) | (data.visit == '1')))
    rows = source_rows(data, people, first)
    np.testing.assert_array_equal(rows, check)
    train = data.subset(rows)
    sessions = Counter(f'{p}|{v}' for p, v in zip(train.person, train.visit, strict=True))
    path = run / fold
    m, p = verify_artifact(path, public=PUBLIC_SHA, transfer=external['weights_sha'], external=external)
    require((m['people'], m['first_session_only'], m['steps'], m['epochs']) == (people, first, 0, 20), fold)
    require(p['names'] == data.names, 'checkpoint montage differs')
    receipt = dict(weights_sha=m['weights_sha'], source_sha=sha(path / 'submission.py'),
                   train_sha=sha(path / 'training.json'))
    replays = 0
    if fold != 'production':
        fitlog = next(run.glob(f'csA-fit-manas-64887774_{jobs.index(job)}.out'))
        require(m['weights_sha'] in fitlog.read_text() and 'public epoch 20/20' in fitlog.read_text(), fitlog)
        for stage in ('confirm', 'extra'):
            receipts = sorted((lane / stage / fold / 'ba_terminal').glob('*.json'))
            require(len(receipts) == 4, f'{stage}/{fold}: incomplete replay receipts')
            for replay in receipts:
                require(read_json(replay)['partner'] == receipt, replay)
                replays += 1
    rng = np.random.default_rng(m['seed'])
    probs = sample_weights(train)
    session_rows = tuple(train.sessions())
    used = np.zeros(len(train.y), dtype=int)
    for _ in range(m['epochs']):
        for group in session_rows:
            rng.permutation(group)
        for _ in range(len(rows) // BATCH):
            draws = rng.choice(len(rows), BATCH, p=probs)
            np.add.at(used, draws, 1)
            advance_augmentation(rng, len(session_rows) ** 2)
    require(all(used[group].sum() for group in session_rows), 'a listed session supplied no gradient samples')
    mod = load_source(path / 'submission.py')
    net = mod.Decoder('manas', p['names'], p['model']['positions'])
    net.load_state_dict(p['model'], strict=True)
    device = torch.device('cuda')
    net.to(device)
    recomputed = bank(net, train, rng, device, BATCH, 'A1')
    stored = p['bank']
    require(set(stored) == set(recomputed), 'unchecked stored statistics')
    stats = {}
    for key, value in stored.items():
        before, after = np.asarray(value), np.asarray(recomputed[key])
        if key in ('covariance', 'gallery', 'identity_scale', 'identity_radius', 'enrolled'):
            np.testing.assert_allclose(after, before, rtol=3e-6, atol=1e-6, err_msg=key)
        else:
            np.testing.assert_allclose(after, before, rtol=2e-3, atol=1e-3, err_msg=key)
        stats[key] = dict(shape=before.shape, sha256=sha256(before.tobytes()).hexdigest(),
                          max_abs=float(np.max(np.abs(after.astype(float) - before))))
    return dict(fold=fold, people=people, public_windows=len(rows), sessions=dict(sorted(sessions.items())),
                first_session_only=first, transfer_sha=external['weights_sha'], lane_receipts=replays,
                rows_sha=sha256(rows.astype('<i8').tobytes()).hexdigest(), stats=stats, **receipt,
                gradient_draws=int(used.sum()), distinct_gradient_windows=int(np.count_nonzero(used)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('team_a', type=Path)
    p.add_argument('public', type=Path)
    p.add_argument('lane', type=Path)
    p.add_argument('phase')
    p.add_argument('out', type=Path)
    a = p.parse_args()
    sys.path.insert(0, str(a.team_a / 'cleanslate/teamA'))
    from data import sha

    require(sha(a.public) == PUBLIC_SHA, 'public export differs')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    selected = a.team_a / 'b6-a1-manas'
    niv = a.team_a / 'b6-a1-niv-manas'
    if a.phase == 'external':
        logs = ('csA-ext-manas-64887773_*.out', 'csA-nivext-manas-64887777_*.out')
        report = {scope: external_fit(folder / 'external', a.public, a.team_a / 'external',
                                      next(folder.glob(log)), scope == 'niv')
                  for scope, folder, log in zip(('production', 'niv'), (selected, niv), logs, strict=True)}
    else:
        experiment = niv if a.phase == 'niv-production' else selected
        transfer, _ = verify_artifact(experiment / 'external')
        report = audit_fold('production' if a.phase == 'niv-production' else a.phase,
                            a.public, experiment, a.lane, transfer)
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(a.phase, 'PASS', flush=True)


if __name__ == '__main__':
    main()
