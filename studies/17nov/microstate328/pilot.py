"""Guard first, then source-only microstate pilot; no outer-person labels (§328)."""
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from representation import K, features, fit, raw287, summarize, tokenize

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026')
DATA = PROJECT / 'data/proteus_windows.npz'
DATA_SHA = '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
LABELS = Path('/mnt/lustre/work/gehler/gfh527/ni26_s306_expert_cpu_v2_20261116/source_labels.py')
LABEL_SHA = '35edab56fe754e6c77a81f54f2010baca7e0912af55af9564043b2a8bde8c4dc'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    started = time.monotonic()
    if sha(DATA) != DATA_SHA or sha(LABELS) != LABEL_SHA: raise ValueError('data/selected-label reader changed')
    for line in (ROOT / 'source.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if sha(ROOT / name) != expected: raise ValueError('immutable source changed')
    spec = importlib.util.spec_from_file_location('selected_source_labels', LABELS)
    reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
    sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / '2026-competition')]
    import protocol
    if Path(protocol.__file__).resolve() != ROOT / 'experiments/protocol.py': raise ValueError('metric protocol import differs')
    with np.load(DATA, allow_pickle=False) as z:
        people = np.array([str(s).split('/')[-1] for s in z['subject']])
        source = np.flatnonzero(np.isin(people, [f'P{i:02d}' for i in range(4, 11)]))
        query = np.flatnonzero(np.isin(people, ['P02', 'P03']))
        rows = np.r_[source, query]; raw = z['X'][rows]
        stream = np.char.add(np.char.add(z['subject'][query], '|'), z['session'][query])
    if set(people[source]) != {f'P{i:02d}' for i in range(4, 11)} or np.any(people[rows] == 'P01'):
        raise ValueError('source/query exclusion failed')
    n = len(source); scale = float(np.sqrt(np.mean(raw[:n].astype(np.float64)**2)))
    centers, random = fit(raw[:n], people[source])
    if not np.allclose(np.linalg.norm(centers, axis=1), 1): raise ValueError('nonunit centroids')
    small = raw[:3]; tokens = tokenize(small, centers)
    if not np.array_equal(tokens, tokenize(-small, centers)) or not np.array_equal(tokens, tokenize(2*small, centers)):
        raise ValueError('polarity/amplitude invariance failed')
    banks = features(raw, centers, random, scale)
    if np.array_equal(banks['ordered'][:, 287+K:], banks['shuffled'][:, 287+K:]): raise ValueError('transition shuffle inactive')
    if np.array_equal(raw287(small, scale), raw287(2*small, scale)): raise ValueError('raw amplitude branch inactive')
    changed = small.copy(); changed[:, 0, -1] += 10 * scale
    if np.array_equal(features(small, centers, random, scale)['ordered'], features(changed, centers, random, scale)['ordered']):
        raise ValueError('last sample ignored')
    for name, value in banks.items():
        if not np.isfinite(value).all(): raise ValueError('nonfinite feature')
        if name in ('ordered', 'shuffled', 'random') and not np.array_equal(value[:, :287], banks['raw287']):
            raise ValueError('raw block changed across controls')
    # Each call takes one current window only; batching, prefix and interleaving must agree.
    for index in (0, 1, 2):
        separate = features(small[index:index+1], centers, random, scale)
        together = features(small, centers, random, scale)
        for name in separate:
            if not np.array_equal(separate[name][0], together[name][index]): raise ValueError('row/prefix independence failed')
    guard = dict(passed=True, labels_opened=False, P01_excluded=True, fit_people=sorted(set(people[source])),
        source_rows_sha256=digest(source), query_rows_sha256=digest(query), codebook_sha256=digest(centers),
        data_sha256=DATA_SHA, source_sha256=sha(ROOT/'source.sha256'), seconds=time.monotonic()-started)
    (ROOT / 'guard.json').write_text(json.dumps(guard, indent=2))
    yfit = reader.read_selected(DATA, 'y', source, len(people))
    models = {}
    for name, bank in banks.items():
        m = make_pipeline(StandardScaler(), LogisticRegression(C=.01, max_iter=2000, class_weight='balanced'))
        m.fit(bank[:n], yfit)
        if int(m[-1].n_iter_.max()) >= 2000: raise ValueError('head not converged')
        models[name] = m
    # Query labels opened only after complete no-score guard and all source fits.
    yquery = reader.read_selected(DATA, 'y', query, len(people))
    interface = reader.read_selected(DATA, 'interface', query, len(people))
    results = {}
    for name, m in models.items():
        pred = m.predict(banks[name][n:]); scores = {}
        for person in ('P02', 'P03'):
            mask = people[query] == person
            scores[person] = protocol.score(yquery[mask], pred[mask], dict(stream=stream, interface=interface), mask)
        results[name] = dict(people=scores, mean_person_cell_ba=float(np.mean([v['balanced_accuracy_cell_mean'] for v in scores.values()])),
            feature_sha256=digest(banks[name]), dimensions=banks[name].shape[1])
    sample = small[:1]; times=[]
    for _ in range(35):
        t=time.perf_counter();models['ordered'].predict_proba(features(sample, centers, random, scale)['ordered']);times.append(time.perf_counter()-t)
    report=dict(source_only=True, P01_labels_opened=False, previously_exposed_queries=['P02','P03'],
        results=results, guard=guard, whole_model_b1_ms_median=float(1000*np.median(times[5:])),
        timing_scope='all six feature banks plus ordered head, CPU B1;conservative pilot path',
        total_seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256=sha(ROOT/'source.sha256'), label_reader_sha256=LABEL_SHA,
        limitation='two exposed source people;no outer/current-system/returning/sealed gain')
    with (ROOT/'result.json').open('x') as f:json.dump(report,f,indent=2)
    np.savez(ROOT/'artifacts.npz', source_rows=source,query_rows=query,centers=centers,random=random,scale=scale)
    print(json.dumps(report),flush=True)


if __name__ == '__main__': main()
