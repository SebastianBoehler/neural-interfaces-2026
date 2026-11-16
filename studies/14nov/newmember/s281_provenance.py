"""Registered §281 recipes and input checksums."""
import hashlib
import json
import os
from pathlib import Path

RECIPES = {
    'control': ('none', 'none', False),
    'S1': ('fb_coral', 'none', False),
    'S2': ('fb_am', 'none', False),
    'S3': ('fb_coral,fb_am', 'none', False),
    'S4': ('none', 'fb_coral', False),
    'S5': ('none', 'fb_am', False),
    'zero_coral': ('fb_coral', 'none', True),
}


def validate_recipe(name, add, swap, zero, seeds):
    if zero not in ('', '0', '1'):
        raise ValueError(f'invalid ZERO={zero!r}')
    got = (add, swap, zero == '1')
    if name not in RECIPES or got != RECIPES[name]:
        raise ValueError(f'{name}: recipe {got} does not match {RECIPES.get(name)}')
    if seeds != '0,1,2,3,4,5':
        raise ValueError(f'{name}: expected six ordered seeds, got {seeds!r}')
    return {'name': name, 'add': add, 'swap': swap, 'zero': zero == '1', 'seeds': seeds}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class InputTracker:
    def __init__(self):
        self.files = {}

    def track(self, file):
        if not isinstance(file, (str, os.PathLike)):
            raise ValueError('cannot record a non-path input')
        path = str(Path(file).resolve(strict=True))
        if path not in self.files:
            self.files[path] = sha256(path)

    def manifest(self):
        for path, expected in self.files.items():
            if sha256(path) != expected:
                raise ValueError(f'input changed during run: {path}')
        return dict(sorted(self.files.items()))


def write_manifest(path, manifest):
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(manifest, f, sort_keys=True, indent=2)
        f.write('\n')
