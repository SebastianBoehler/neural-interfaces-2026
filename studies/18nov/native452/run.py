"""Marshal raw EEG and frozen tensors; the C++ library performs every decoder operation."""
import ctypes
import json
from pathlib import Path
import sys

import numpy as np
import torch

import train
import score379 as ref

HISTORY = ref.WORK/'ni26_history443_20261118'
REFERENCE_SHA = '8ff0d3af5bc1fbdd31bf2bd958a659f4f072772c11aa3583d72430283d1bfdb0'
ROOTS = {379: ('439', 'b6a6397455730088ce45a041b06510abe9ab57e7266d8ecd62e6e84060e8c8f3'),
         1441: ('441', 'a82db5447b9d7d6844786f76d06af271c02f0055a04455f36738e13149825518')}
KEYS = ('spatial.weight', 'temporal.weight', 'write.0.weight', 'write.0.bias',
        'read.0.weight', 'read.0.bias', 'read.2.weight', 'read.2.bias',
        'head.0.weight', 'head.0.bias', 'head.2.weight', 'head.2.bias', 'mean', 'second')


def native():
    c = ctypes
    lib = c.CDLL(str(Path('decoder.so').resolve()))
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    decode = lib.decode452
    decode.argtypes = (f32, c.c_int, f32, c.c_int, c.c_float, f32, c.c_char_p, c.c_int)
    decode.restype = c.c_int

    def execute(w, x, rms):
        logits = np.empty((len(x), 2, 3), np.float32)
        error = c.create_string_buffer(4096)
        if decode(w, w.size, np.ascontiguousarray(x), len(x), rms, logits, error, len(error)):
            raise RuntimeError(error.value.decode())
        return logits
    return execute


def compare(seed, person, smoke=False):
    section, expected = ROOTS[seed]
    folder = ref.WORK/f'ni26_gain{section}_20261118/coherent'
    record = json.loads((folder/(person+'.json')).read_text())
    proof = json.loads((HISTORY/f'{seed}_{person}.json').read_text())
    archive = HISTORY/f'{seed}_{person}.npz'
    checkpoint = folder/(person+'_memory.pt')
    original = folder/(person+'.npz')
    assert proof['source'] == ref.source_manifest(HISTORY/'source.sha256') == REFERENCE_SHA
    assert record['source_sha256'] == ref.source_manifest(folder/'source.sha256') == expected
    assert record['checkpoints']['memory'] == ref.sha(checkpoint) == proof['checkpoint']
    assert record['outputs_sha256'] == ref.sha(original) == proof['original']
    assert ref.sha(archive) == proof['output']
    assert ref.sha(train.DATA) == ref.DATA_SHA and ref.sha(train.PERM) == record['order_sha256'] == ref.ORDER_SHA
    assert record['held'] == person and set(record['train_people']) == set(ref.PEOPLE)-{person}
    net = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert set(net) == set(KEYS) and all(v.dtype == torch.float32 for v in net.values())
    weights = np.concatenate([net[k].numpy().ravel() for k in KEYS])
    device = torch.cuda.get_device_name()
    assert 'A100' in device
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with np.load(train.DATA) as z:
        x = np.asarray(z['X'], np.float32)
        people = np.array([s.split('/')[-1] for s in z['subject']])
        sessions = z['session']
    fit, test = np.flatnonzero(people != person), np.flatnonzero(people == person)
    assert train.digest(fit) == record['train_rows_sha256']
    rms = torch.from_numpy(x).cuda()[fit].square().mean().sqrt().item()
    decode = native()
    report = dict(seed=seed, person=person, checkpoint=proof['checkpoint'], reference=proof['output'],
                  original=proof['original'], source=ref.sha(Path('source.sha256')),
                  library=ref.sha(Path('decoder.so')), device=device, torch=torch.__version__,
                  rms=rms, decisions=0, sessions=0, reset_errors=0, max_abs=0., worst=None,
                  changed_by_history=0, mismatches=[])
    with np.load(archive) as refbank, np.load(original) as old, np.load(train.PERM) as perms:
        assert np.array_equal(refbank['index'], test) and np.array_equal(old['index'], test)
        for oi, order in enumerate(ref.ORDERS):
            if smoke and order != 'full0':
                continue
            perm = perms[f'{order[:-1]}|{order[-1]}|all_loso']
            assert np.array_equal(np.sort(perm), np.arange(len(x)))
            for s in np.unique(sessions[test]):
                rows = perm[(people[perm] == person) & (sessions[perm] == s)]
                if smoke:
                    rows = rows[:24]
                pos = np.searchsorted(test, rows)
                assert np.array_equal(test[pos], rows)
                logits = decode(weights, x[rows], rms)
                assert np.isfinite(logits).all()
                report['sessions'] += 1
                report['reset_errors'] += int(not np.array_equal(logits[0,0], logits[0,1]))
                report['changed_by_history'] += int((logits[:,0].argmax(1) != logits[:,1].argmax(1)).sum())
                for arm, name in enumerate(('memory', 'blank')):
                    truth = refbank[name][oi, pos]
                    preds = truth.argmax(1)
                    assert np.array_equal(preds, old[f'{name}|{order}'][pos])
                    difference = np.abs(logits[:,arm] - truth)
                    idx, cls = np.unravel_index(difference.argmax(), difference.shape)
                    if difference[idx, cls] > report['max_abs']:
                        report.update(max_abs=float(difference[idx,cls]), worst=[order,str(s),int(rows[idx]),name,int(cls)])
                    for idx in np.flatnonzero(logits[:,arm].argmax(1) != preds):
                        report['mismatches'].append([order, str(s), int(rows[idx]), name,
                                                     logits[idx,arm].tolist(), truth[idx].tolist()])
                    report['decisions'] += len(rows)
    report['pass'] = not report['mismatches'] and not report['reset_errors'] and report['max_abs'] <= 2e-4
    if not smoke:
        assert report['decisions'] == 2 * len(test) * len(ref.ORDERS)
    path = Path('smoke.json' if smoke else f'{seed}_{person}.json')
    path.write_text(json.dumps(report, allow_nan=False))
    print(path, report['decisions'], report['max_abs'], report['pass'], flush=True)
    if smoke:
        assert report['pass'] and report['changed_by_history'] > 0


if __name__ == '__main__':
    compare(int(sys.argv[1]), sys.argv[2], 'smoke' in sys.argv[3:])
