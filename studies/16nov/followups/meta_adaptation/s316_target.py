"""Explicit common-anchor §316 target; preserve the original streaming code."""
from pathlib import Path


def patch_source(source):
    replacements = [
        ('from bridge import Native', 'from cpu_crossfit import CPUReference as Native\nfrom s316_anchor import load_common_anchor'),
        ("device = torch.device('cuda:0')", "device = torch.device('cpu')"),
        ("    labels = np.ascontiguousarray(d['y'][train_rows],dtype=np.int64)\n"
         "    native = Native(); start = time.monotonic()\n"
         "    train = native.standardize(X, mean, scale)\n"
         "    theta,loss,grad,calls = fit_native(native,train,labels)",
         "    native = Native(); start = time.monotonic()\n"
         "    theta,loss,grad,calls = load_common_anchor(shard, refmeta, cache_hashes,\n"
         "        sha(ROOT/'data/proteus_windows.npz'), sha(ROOT/'runs/shuffle_perms.npz'),\n"
         "        os.environ['S307_CACHE_LOCK_SHA'])"),
        ("    meta = {'held':held,", "    meta = {'anchor_policy':'s305_common','held':held,"),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f'expected one §307 source anchor: {old[:100]}')
        source = source.replace(old, new, 1)
    return source


if __name__ == '__main__':
    path = Path(__file__).with_name('target.py')
    exec(compile(patch_source(path.read_text()), str(path), 'exec'),
         {'__name__': '__main__', '__file__': str(path)})
