import numpy as np, csv, glob, re, collections
z = np.load('data/proteus_windows.npz', allow_pickle=True)
subj = np.array([s.split('/')[-1] for s in z['subject']]); ses = z['session'].astype(str); task = z['task'].astype(str); run = z['run']; onset = z['onset']; y = z['y']; itf = z['interface'].astype(str)
root = 'data/neural_compet/Dreyer2026Proteus/download/nm000290'
n_ok = n_rows = 0; codes = collections.Counter(); bad = []; lat = collections.defaultdict(list)
for f in sorted(glob.glob(f'{root}/sub-*/ses-*/eeg/*_events.tsv')):
    m = re.search(r'sub-(P\d+)_ses-(\d)_task-(\w+)_run-(\d+)_events', f)
    p, s, t, r = m.group(1), m.group(2), m.group(3), int(m.group(4))
    rows = list(csv.DictReader(open(f), delimiter='\t'))
    cue = [(float(e['onset']), str(e.get('value', e.get('trial_type')))) for e in rows if str(e.get('value', '')) in ('33024', '33025', '33026')]
    fb = [float(e['onset']) for e in rows if str(e.get('value', '')) == '781']
    sel = (subj == p) & (ses == s) & (task == t) & (run == r)
    if not cue:
        assert sel.sum() == 0, f; continue
    o = np.argsort(onset[sel]); on, yy = onset[sel][o], y[sel][o]
    n_rows += sel.sum()
    if len(cue) != sel.sum(): bad.append((f, len(cue), int(sel.sum()))); continue
    d = on - np.array([c[0] for c in cue])
    lab = np.array([{'33024': 0, '33025': 1, '33026': 2}[c[1]] for c in cue])
    n_ok += int(((np.abs(d - 1.0) < 2e-3) & (lab == yy)).sum())
    for c, k in zip(cue, yy): codes[(c[1], int(k))] += 1
    if fb and len(fb) == len(cue): lat[t].extend((np.array(fb) - np.array([c[0] for c in cue])).tolist())
print('rows checked', n_rows, 'rows with window start = cue + 1.000 s and label = code order', n_ok, 'mismatched runs', bad)
print('code -> label counts', dict(codes))
for t, v in lat.items(): print('feedback-start (781) minus cue, task', t, 'median %.2f s, min %.2f, max %.2f, n %d' % (np.median(v), np.min(v), np.max(v), len(v)))
print('interface from task name:', {t: set(itf[task == t]) for t in sorted(set(task))})
