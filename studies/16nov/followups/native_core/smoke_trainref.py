"""No-score native training-reference component parity."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_ref import test_training_reference_and_features, test_fold_prior_excludes_held_person  # noqa: E402

test_training_reference_and_features()
test_fold_prior_excludes_held_person()
print('native whole-session reference/feature parity passed; no score', flush=True)
