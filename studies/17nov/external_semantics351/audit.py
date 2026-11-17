"""Qualify existing ShinWG/Scherer metadata without fitting or EEG sample decoding."""
import argparse
import csv
import gzip
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from mat_metadata import read

SCHERER_CHANNELS = ('AFz F7 F3 Fz F4 F8 FC3 FCz FC4 T7 C3 Cz C4 T8 CP3 CPz CP4 P7 P5 '
                    'P3 P1 Pz P2 P4 P6 P8 PO3 PO4 O1 O2').split()
LABELS = {1: 2, 2: 1, 4: 0}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def metadata(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != 'X'}


def shin(root, out, smoke):
    path = root / 'data/shin2018/windows.npz'
    d = metadata(path)
    assert d['classes'].tolist() == ['baseline', 'WG']
    assert len(d['y']) == 3120 and np.bincount(d['y']).tolist() == [1560, 1560]
    common, records, rows, keys = set(d['ch_names']), [], [], set()
    for person in range(1, 2 if smoke else 27):
        for run in range(1, 2 if smoke else 4):
            directory = root / f'data/shin2018/raw/VP{person:03d}'
            hdr = directory / f'word{run}.vhdr'
            mrk = directory / f'word{run}.vmrk'
            text = hdr.read_text()
            fs = 1e6 / float(re.search(r'^SamplingInterval=(\d+)', text, re.M)[1])
            channels = re.findall(r'^Ch\d+=([^,]*),[^,]*,([^,]*),([^,\r\n]*)', text, re.M)
            assert fs == 1000 and len(channels) == 30
            assert {c[2] for c in channels} == {'µV'}
            names = {c[0] for c in channels}
            common &= names
            cues = []
            for line in mrk.read_text().splitlines():
                if line.startswith('Mk'):
                    parts = line.split('=', 1)[1].split(',')
                    if parts[0] == 'Stimulus' and parts[1] in ('S 16', 'S 32'):
                        cues.append(((int(parts[2]) - 1) / fs, int(parts[1] == 'S 16')))
            assert len(cues) == 20 and sum(c for _, c in cues) == 10
            ids = np.flatnonzero((d['subject'] == f'Shin2018/{person}') & (d['run'] == str(run)))
            assert len(ids) == 40
            expected = np.array([t + offset for t, _ in cues for offset in (0, 4)])
            expected_y = np.array([y for _, y in cues for _ in range(2)])
            assert np.allclose(d['onset'][ids], expected, atol=1e-9, rtol=0)
            assert np.array_equal(d['y'][ids], expected_y)
            assert np.array_equal(d['trial'][ids], np.repeat(np.arange(20), 2))
            assert np.array_equal(d['win'][ids], np.tile([0, 1], 20))
            assert np.array_equal(d['block'][ids], np.full(40, run))
            for i in ids:
                key = (str(d['subject'][i]), run, int(d['trial'][i]), int(d['win'][i]))
                assert key not in keys
                keys.add(key)
                rows.append([int(i), *key, int(d['y'][i]), float(d['onset'][i]),
                             float(d['onset'][i] + 1), float(d['onset'][i] + 5)])
            records.append({'person': person, 'run': run, 'header_sha256': sha(hdr),
                            'marker_sha256': sha(mrk), 'marker_count': len(cues),
                            'data_reference_typo': f'=wor{run}.' in text})
    if not smoke:
        assert len(keys) == 3120 and common == set(d['recorded'])
    write_rows(out / 'shin_rows.tsv.gz',
               ['row', 'person', 'run', 'trial', 'crop', 'class', 'stored_onset_s', 'crop_start_s', 'crop_end_s'], rows)
    return {'source_sha256': sha(path), 'rows': len(rows), 'recordings': records,
            'native_target_channels': sorted(common), 'EEG_sampling_hz': 1000,
            'stored_onset_to_actual_crop_s': 1, 'unique_row_identity': True}


def write_rows(path, fields, rows):
    with gzip.open(path, 'wt') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(fields)
        writer.writerows(rows)


def scherer(root, out, smoke):
    path = root / 'data/scherer2015/windows.npz'
    d = metadata(path)
    assert len(d['y']) == 2130 and set(d['y']) == {0, 1, 2}
    assert np.array_equal(d['run'], np.zeros(len(d['run'])))
    native = [c for c in d['ch_names'] if c in SCHERER_CHANNELS]
    assert len(native) == 19 and set(d['ch_names']) - set(native) == set(d['interpolated'])
    records, rows, seen, groups = [], [], set(), []
    for person in ('A' if smoke else 'ACDEFGHJL'):
        raw = root / f'data/scherer2015/raw/{person}.mat'
        sessions = read(raw)
        assert len(sessions) == 2
        for session in sessions:
            assert session['fs'] == 256 and session['X']['shape'][1] == 30
            assert session['X']['numeric_payload_skipped']
            assert session['classes'] == ['WORD', 'SUB', 'NAV', 'HAND', 'FEET']
            trial, y, day = session['trial'], session['y'], int(session['session'])
            assert len(trial) == len(y) and len(trial) % 25 == 0
            assert (np.diff(trial) > 0).all()
            n = len(trial)
            boundary = np.arange(25, n, 25)
            gaps = np.diff(trial.astype(float)) / 256
            regular = np.delete(gaps, boundary - 1)
            assert regular.min() >= 12.49 and regular.max() <= 13.51
            assert (gaps[boundary - 1] > 20).all()
            run_counts = []
            for r in range(n // 25):
                counts = np.bincount(y[r * 25:(r + 1) * 25].astype(int), minlength=6)[1:]
                assert counts.tolist() == [5] * 5
                run_counts.append(counts.tolist())
                groups.append((person, day, r + 1))
            ids = np.flatnonzero((d['subject'] == f'Scherer2015/{person}') & (d['session'] == str(day)))
            selected = np.flatnonzero(np.isin(y, list(LABELS)))
            cues = trial[selected] / 256 + 3
            assert np.array_equal(d['onset'][ids], cues)
            assert np.array_equal(d['y'][ids], [LABELS[int(c)] for c in y[selected]])
            for i, source_trial in zip(ids, selected):
                key = (person, day, int(source_trial))
                assert key not in seen
                seen.add(key)
                # Pinned exporter uses trial/fs; preserve its existing convention explicitly.
                rows.append([int(i), person, day, int(source_trial // 25 + 1), int(source_trial),
                             int(source_trial % 25), int(y[source_trial]), int(d['y'][i]),
                             float(d['onset'][i]), float(d['onset'][i] + 1), float(d['onset'][i] + 5)])
            records.append({'person': person, 'session': day, 'source_trials_all_five_classes': n,
                            'exported_rows': len(ids), 'runs': len(run_counts),
                            'native_shape': session['X']['shape'], 'raw_sha256': sha(raw),
                            'within_run_gap_s': [float(regular.min()), float(regular.max())],
                            'between_run_gap_s': [float(v) for v in gaps[boundary - 1]],
                            'class_counts_by_run': run_counts})
    if not smoke:
        assert len(seen) == 2130
    write_rows(out / 'scherer_rows.tsv.gz', ['row', 'person', 'session', 'run', 'source_trial',
               'within_run_trial', 'raw_class', 'export_class', 'cue_s', 'crop_start_s', 'crop_end_s'], rows)
    return {'source_sha256': sha(path), 'rows': len(rows), 'sessions': records,
            'runs': len(groups), 'native_target_channels': native, 'EEG_sampling_hz': 256,
            'unique_row_identity': True, 'pair_WORD_SUB_rows': sum(row[7] in (1, 2) for row in rows),
            'run_recovery': '25 consecutive trials, five of each class; all internal gaps 12.49–13.51s, boundary gaps >20s'}


def main(args):
    t, cpu = time.monotonic(), time.process_time()
    args.out.mkdir(parents=True, exist_ok=False)
    result = {'scope': 'metadata only; EEG numeric payload never decoded',
              'shin': shin(args.root, args.out, args.smoke),
              'scherer': scherer(args.root, args.out, args.smoke),
              'source_hashes': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}}
    result['wall_s'], result['cpu_s'] = time.monotonic() - t, time.process_time() - cpu
    (args.out / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'shin_rows': result['shin']['rows'], 'scherer_rows': result['scherer']['rows'],
                      'scherer_runs': result['scherer']['runs'], 'wall_s': result['wall_s'], 'cpu_s': result['cpu_s']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    main(p.parse_args())
