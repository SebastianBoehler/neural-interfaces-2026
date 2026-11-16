import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / 'studies/16nov/followups/native_core'),
                str(ROOT / 'studies/16nov/followups/geometry')]
from score_chain import patch_source  # noqa: E402


def test_existing_chain_scorer_contains_registered_matched_contrasts():
    path = ROOT / 'studies/16nov/followups/native_core/score.py'
    spec = importlib.util.spec_from_file_location('native_score_313', path)
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    original = (ROOT / 'studies/16nov/followups/geometry/score.py').read_text()
    patched = patch_source(original, score)
    compile(patched, '<s313-chain-score>', 'exec')
    assert "('IMAG','REAL')" in patched
    assert "('BOTH','REAL')" in patched
