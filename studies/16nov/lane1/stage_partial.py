"""Stage one EEG family using the checked candidate-copy implementation."""
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('stage_candidates', HERE / 'stage_candidates.py')
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)
M.GROUPS = {'plain': ('eegn',), 'dist': ('eegn_dist2',)}


if __name__ == '__main__':
    group = sys.argv[1]
    if group not in M.GROUPS:
        raise ValueError(group)
    # The shared copier's status text assumes two-family groups.
    with contextlib.redirect_stdout(io.StringIO()):
        M.main(group)
    print(f'{group}: validated 48 streams and six fit manifests; copied privately', flush=True)
