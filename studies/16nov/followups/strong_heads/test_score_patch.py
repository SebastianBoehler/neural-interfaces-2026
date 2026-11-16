import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from score_308 import patch_source


def test_existing_scorer_keeps_seed_outputs_before_averaging():
    source = (ROOT / 'studies/16nov/followups/geometry/score.py').read_text()
    import importlib.util
    spec = importlib.util.spec_from_file_location('native_score', ROOT / 'studies/16nov/followups/native_core/score.py')
    sys.path.insert(0, str(ROOT / 'studies/16nov/followups/native_core'))
    sys.path.insert(0, str(ROOT / 'studies/16nov/followups/geometry'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = patch_source(source, mod)
    compile(result, '<s308-score>', 'exec')
    assert "for arm in ('control', *ARMS[1:]):" in result
    assert "metric[arm + '0', order] + metric[arm + '1', order]" in result
    assert "('Stanh','Slinear')" in result
