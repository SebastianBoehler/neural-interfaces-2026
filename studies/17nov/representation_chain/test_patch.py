"""Narrow adapter checks: original anchors, recipe identity and fixed ensemble sharing."""
import ast
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from config import BASES, GROUPS, recipes
from patch import main

REPO = Path(__file__).resolve().parents[3]


def check(group):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        lane = root / 'studies/16nov/lane1'; lane.mkdir(parents=True)
        shutil.copy(REPO / 'studies/16nov/lane1/patch_private.py', lane)
        target = root / 'studies/14nov/newmember'; target.mkdir(parents=True)
        for name in ('s281_provenance.py', 'run_s.py', 's281_cp.py', 'score_s_v66p.py', 'run_s_v66.py'):
            shutil.copy(REPO / 'studies/14nov/newmember' / name, target)
        main(group, root)
        spec = importlib.util.spec_from_file_location('patched315_provenance', target / 's281_provenance.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for arm, (add, swap, zero) in recipes(group).items():
            module.validate_recipe(arm, add, swap, str(int(zero)), '0,1')
        try:
            module.validate_recipe('control', 'none', 'none', '0', '0,1,2,3,4,5')
        except ValueError:
            pass
        else:
            raise AssertionError('six synthetic seeds accepted')
        source = ast.parse((target / 'run_s.py').read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'stream')
        family = GROUPS[group]['families'][0]
        p0, p1 = np.array([[.2, .3, .5], [.1, .8, .1]]), np.array([[.4, .5, .1], [.3, .2, .5]])
        average = (p0 + p1) / 2
        path = root / f'ttt18within0_{family}_base-all_loso-ensemble2.npz'
        np.savez(path, index=np.arange(2), proba=average, metadata=json.dumps(dict(
            seeds=[0, 1], operation='arithmetic_probability_mean', family=family, order='within0')))
        env = dict(np=np, STREAM={}, BS=dict.fromkeys(BASES), NM_SEEDS=[0, 1], NM_DIR=str(root),
                   TAG='ttt18within0', p0='all_loso', N=2, O='within0', norm=lambda x: x)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<stream>', 'exec'), env)
        arrays = [env['stream'](family, b) for b in BASES]
        assert all(np.array_equal(x, average) and x is arrays[0] for x in arrays)
        scorer = (target / 'score_s_v66p.py').read_text()
        assert 'ensemble2.npz' in scorer and 'FAMILIES315' in scorer
        assert "recipe.get('seeds') != '0,1'" in scorer
    print(f'{group}: original patch anchors, two-seed identity and fixed six-base member checks passed')


if __name__ == '__main__':
    for group in GROUPS:
        check(group)
