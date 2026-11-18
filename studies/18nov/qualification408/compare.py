"""Check a matched pair and save actual wall seconds without EEG or labels."""
import json
from pathlib import Path
import sys


def compare(rep):
    def read(arm):
        return json.loads(Path(f'{arm}-{rep}.json').read_text())
    a, b = read('stock'), read('pruned')
    assert a.keys() == b.keys()
    rows = {}
    for name, stock in a.items():
        pruned = b[name]
        for key in stock.keys() - {'views', 'seconds'}:
            assert stock[key] == pruned[key], (name, key)
        if rep != '-1':
            assert stock['views'] > pruned['views']
        rows[name] = {key: [stock[key], pruned[key]] for key in ('views', 'seconds')}
        rows[name].update({key: stock[key] for key in ('activity', 'latch', 'rows_sha')})
        if 'posterior_sha' in stock:
            rows[name]['posterior_sha'] = stock['posterior_sha']
    wall = [float(Path(f'{arm}-{rep}.seconds').read_text()) for arm in ('stock', 'pruned')]
    Path(f'pair-{rep}.json').write_text(json.dumps(dict(streams=rows, process_seconds=wall), indent=2))


if __name__ == '__main__':
    compare(sys.argv[1])
