"""Change one frozen CP scheduling predicate, with measured refit activity."""
import inspect
import textwrap
import time

import numpy as np

ARMS = ('frozen', 'skip', 'row_budget', 'recycle')
EXTRA = frozenset((90, 110, 130))
FIRST = 80


def scheduled(arm, i):
    if arm == 'frozen':
        return i >= 20 and i % 20 == 0
    if arm in ARMS[1:]:
        return ((i >= FIRST and i % 20 == 0)
                or (arm == 'row_budget' and i == 110)
                or (arm == 'recycle' and i in EXTRA))
    raise ValueError(f'unknown schedule {arm}')


def factories(parent, config):
    if (config['v6_cp_t0'], config['v6_cp_k'], config['v6_cp_w0']) != (20, 20, 90):
        raise ValueError('original package CP schedule changed')
    old = 'if i >= self.t0 and (i - self.t0) % self.k == 0:'
    src = textwrap.dedent(inspect.getsource(parent.step))
    if src.count(old) != 1:
        raise ValueError('original CP scheduling predicate missing')
    scope = dict(parent.step.__globals__)
    exec(compile(src.replace(old, 'if scheduled(self.arm, i):'), '<cp-refit-gate365>', 'exec'),
         dict(scope, scheduled=scheduled), scope)

    class Frozen(parent):
        arm = 'frozen'

        def __init__(self, conf):
            self.refit_rows = []
            self.refit_seconds = 0.
            self.selected_work = 0
            super().__init__(conf)

        def select(self, labels, confidence):
            chosen = super().select(labels, confidence)
            self.selected_work += len(chosen)
            return chosen

        def refit(self, prefix):
            if len(self.C) != prefix + 1 or len(self.P) != prefix + 1:
                raise ValueError('CP refit inclusion boundary changed')
            self.refit_rows.append(prefix)
            start = time.perf_counter()
            value = super().refit(prefix)
            self.refit_seconds += time.perf_counter() - start
            return value

        def __del__(self):
            # Replay records stats and fails on every stage error explicitly.
            pass

    class Skip(Frozen):
        arm = 'skip'
        step = scope['step']

    class Recycle(Skip):
        arm = 'recycle'

    class RowBudget(Skip):
        arm = 'row_budget'

    return dict(zip(ARMS, (Frozen, Skip, RowBudget, Recycle), strict=True))


def apply(factory, config, features, raw, filtered, sessions, recognized=None):
    start = time.perf_counter()
    out = np.full_like(raw, np.nan)
    stats = []
    first_fusion = config['v6_cp_w0']
    for rows in sessions:
        model = factory(config)
        model.feats = features.__getitem__
        for i, row in enumerate(rows):
            out[row] = model.step(row, raw[row], filtered[row],
                                  recognized is not None and i >= recognized)
        stop = len(rows) if recognized is None else min(len(rows), recognized)
        expected = [i for i in range(stop) if scheduled(factory.arm, i)]
        if (model.log['err'] or model.dead or model.t != len(rows)
                or model.refit_rows != expected or model.refits != len(expected)
                or model.fused != max(0, stop - first_fusion)):
            raise ValueError(f'CP refit/activity mismatch for {factory.arm}: {model.log}')
        stats.append(dict(windows=len(rows), stopped=model.stopped, refits=model.refit_rows,
                          refit_seconds=model.refit_seconds, fusions=model.fused,
                          single_class=model.nfail, prefix_work=sum(model.refit_rows),
                          selected_work=model.selected_work))
    return out, dict(seconds=time.perf_counter() - start, sessions=stats)


if __name__ == '__main__':
    guards = [[i for i in range(190) if scheduled(arm, i)] for arm in ARMS]
    assert guards == [list(range(20, 190, 20)), list(range(80, 190, 20)),
                      [80, 100, 110, 120, 140, 160, 180],
                      [80, 90, 100, 110, 120, 130, 140, 160, 180]]
    assert list(map(sum, guards)) == [900, 780, 890, 1110]
    assert all(len([i for i in g if i < 63]) == n for g, n in zip(guards, [3, 0, 0, 0]))
    print('OK: registered refits, inactive pre-fusion work, and matched complete count')
