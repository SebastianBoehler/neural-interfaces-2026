"""Aggregate the two genuine time pairs and independently observed posteriors."""
import hashlib
import json
from pathlib import Path
from statistics import median


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect():
    pairs = [json.loads(Path(f'pair-{i}.json').read_text()) for i in range(3)]
    names = list(pairs[0]['streams'])
    assert names == ['P10-3', 'Schirrmeister2017/1', 'Schirrmeister2017/2']
    for name in names:
        ref = pairs[2]['streams'][name]
        for pair in pairs[:2]:
            entry = pair['streams'][name]
            for key in ('activity', 'latch', 'rows_sha', 'views'):
                assert ref[key] == entry[key], (name, key)
        assert len(ref['posterior_sha']) == 64
    savings = {}
    for group, included in [('returning', names[:1]), ('unknown', names[1:])]:
        percent = []
        for pair in pairs[:2]:
            times = [pair['streams'][name]['seconds'] for name in included]
            stock, pruned = map(sum, zip(*times))
            percent.append(100 * (1 - pruned / stock))
        savings[group] = dict(paired_percent=percent, median_percent=median(percent))
    return dict(pairs=pairs, savings=savings, source_sha=digest('source.sha256'),
                private_members_sha=digest('members.sha256'),
                input_pair_sha=[digest(f'pair-{i}.json') for i in range(3)])


if __name__ == '__main__':
    print(json.dumps(collect(), indent=2))
