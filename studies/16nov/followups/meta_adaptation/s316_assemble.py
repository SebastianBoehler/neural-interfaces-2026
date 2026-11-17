"""Require all ten freshly generated §316 common-anchor target folds."""
from pathlib import Path


def patch_source(source):
    old = "m['held']!=held or m['held_accuracy_computed'] is not False"
    new = "m.get('anchor_policy')!='s305_common' or " + old
    if source.count(old) != 1:
        raise ValueError('expected exactly one complete-matrix fold guard')
    return source.replace(old, new, 1)


if __name__ == '__main__':
    path = Path(__file__).with_name('assemble_target.py')
    exec(compile(patch_source(path.read_text()), str(path), 'exec'),
         {'__name__': '__main__', '__file__': str(path)})
