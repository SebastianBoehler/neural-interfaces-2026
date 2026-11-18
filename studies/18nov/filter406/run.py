"""Time finite-prefix pruning against the actual conditional returning package."""
import copy
import inspect
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/mnt/lustre/work/gehler/gfh527')
OWNER = ROOT / 'ni26_js321_20261117_returning_pair_cpu1'
sys.path.insert(0, str(OWNER))
from returning_pair_cell import inputs, fresh_model
from returning_pair_guard import BUNDLE, zero
from package_guard import instrument, frame
from record import load_model
from returning_assets import sha


def variant(model):
    stock = type(model.cf)
    source = textwrap.dedent(inspect.getsource(stock.step))
    anchor = '        self.V.append(self.view(x))'
    if source.count(anchor) != 1:
        raise ValueError('count filter source changed')
    edit = """        if self.latch is None and t >= max(self.thr):
            self.t += 1
            return p
        if self.latch is None:
            self.V.append(self.view(x))"""
    namespace = {}
    exec(compile(source.replace(anchor, edit), '<pruned filter>', 'exec'),
         stock.step.__globals__, namespace)
    new_type = type('FinitePrefixCountFilter', (stock,), {'step': namespace['step']})
    result = copy.deepcopy(model.cf)
    result.__class__ = new_type
    return result


def check_filter(filter_, stream, outputs):
    filter_.reset_state()
    start = time.perf_counter()
    observed = np.stack([filter_.step(x, p) for x, p in stream])
    seconds = time.perf_counter() - start
    if not np.array_equal(observed, outputs) or filter_.dead or filter_.log['err']:
        raise ValueError('filter endpoint changed')
    return seconds


def run(index):
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    fold, order, X, channels, canonical, _, delivery, sessions, _, _, _ = inputs(index)
    package = BUNDLE / fold / 'base'
    model = instrument(load_model(package, channels, 'base406', 'cpu'))
    package_source = package / 'submission.py'
    digest = sha(package_source)
    if digest != '37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f':
        raise ValueError('fold package differs from staged cap250 submission.py')
    proof = json.loads((OWNER / f'cell_{index}/proof.json').read_text())
    previous = OWNER / f'cell_{index}/predictions.npz'
    if sha(previous) != proof['predictions_sha256'] or proof['order'] != order:
        raise ValueError('archived returning endpoint/order changed')
    with np.load(previous) as archive:
        if not np.array_equal(canonical, archive['index']):
            raise ValueError('archived order mismatch')
        scores = archive['base_scores'].copy()
    report = []
    for session in dict.fromkeys(sessions[delivery]):
        rows = delivery[sessions[delivery] == session]
        cur = fresh_model(model)
        zero(cur)
        torch.manual_seed(3211701)
        stock = copy.deepcopy(cur.cf)
        pruned = variant(cur)
        original = cur.cf.step
        stream, outputs = [], []
        def replacement(x, p):
            expected = original(x, p)
            got = pruned.step(x, p)
            if not np.array_equal(expected, got) or pruned.latch != cur.cf.latch:
                raise ValueError('per-call count filter value/latch mismatch')
            stream.append((x.copy(), p.copy()))
            outputs.append(expected.copy())
            return got
        cur.cf.step = replacement
        for row in rows:
            label, score, _ = frame(cur, X[row])
            archived = scores[np.searchsorted(canonical, row)]
            if label != archived.argmax() or not np.array_equal(score, archived):
                raise ValueError(f'archived final endpoint differs at {row}')
        if cur.dead or cur.cp.dead or cur.ft.over:
            raise ValueError('parent failure or time guard')
        pairs = []
        for sequence in ((stock, pruned), (pruned, stock)):
            times = [check_filter(f, stream, outputs) for f in sequence]
            pairs.append(times if sequence[0] is stock else times[::-1])
        report.append(dict(session=str(session), rows=len(rows), latch=stock.latch,
                           stock_views=len(stock.V), pruned_views=len(pruned.V), paired_seconds=pairs))
    Path('/out/result.json').write_text(json.dumps(dict(index=index, fold=fold, order=order,
        archive_sha256=sha(previous), package_source_sha256=digest, sessions=report), indent=2))


if __name__ == '__main__':
    run(int(sys.argv[1]))
