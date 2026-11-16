"""Require §301 full-reference feature audit before the §305 smoke."""
import json
from pathlib import Path

path = Path('/mnt/lustre/work/gehler/gfh527/ni26_s301_native_audit_20261116/runs/s301_audit/head.json')
audit = json.loads(path.read_text())
if (audit['held_accuracy_computed'] is not False or audit['training_feature_rel_l2'] > 1e-6 or
        audit['training_feature_max_abs'] > 1e-4 or not audit['saved_scaler_mean_match'] or not audit['saved_scaler_scale_match'] or
        abs(audit['fits']['strict_native']['objective_on_native_features'] - 0.0015985030536499255) > 1e-5):
    raise ValueError('registered training feature/objective prerequisites failed')
print('registered §305 training-only prerequisites passed; no score', flush=True)
