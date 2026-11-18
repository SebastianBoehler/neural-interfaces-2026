"""Verify deferred and eager whole-worker decisions before reporting times."""
import hashlib
import json
from pathlib import Path
import sys

ROLES = ('recognized','unrecognized')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    assert digest(path) == path.with_suffix('.sha256').read_text().strip()
    value = json.loads(path.read_text())
    assert value['source'] == digest(Path('source.sha256'))
    return value


def pair(stock, deferred):
    a, b = map(read, (stock,deferred))
    assert a['arm'] == 'stock' and b['arm'] == 'deferred'
    assert all(a[k] == b[k] for k in ('gpu','index','mode'))
    exact = ('rows','labels','terminal','work')
    assert a['cold_role'] == b['cold_role']
    assert all(a['cold'][k] == b['cold'][k] for k in exact)
    for role in ROLES:
        x,y = a['records'][role],b['records'][role]
        assert all(x[k] == y[k] for k in exact), role
        assert all(a['warm'][role][k] == b['warm'][role][k] for k in exact), role
        assert x['cov'] > y['cov'] >= 0, role
        if role == 'recognized':
            assert x['cov'] == 10 and y['cov'] == 0
        else:
            assert x['work']['activity']['cp_fused'] > 0 and y['cov'] > 0
    return a,b


if __name__ == '__main__':
    first,second = pair(*sys.argv[1:])
    print('verified whole-worker pair',first['index'],first['mode'],first['gpu'],flush=True)
