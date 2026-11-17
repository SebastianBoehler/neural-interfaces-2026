"""Prepare common PCA input from existing SOURCE features; no head/PCA fitting."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from convert_package import save_tensors

ROOT = Path('/mnt/lustre/work/gehler/gfh527')
FEATURES = ROOT/'ni26_core_training_source_v3_20261117/fixture/reference.pt'
REFERENCE = ROOT/'ni26_js321_20261117_returning_view_fit1/source_view_P04_s23.npz'
PINS = {FEATURES:'aba119c2c6985bc3fef0477c44e2f03b91e8b71e48c90d863c2c3f364bbcff3c',
        REFERENCE:'8f52ab55d4dea7eca9b43b610fad2c645c1f3029372570d05ab71b2187123675'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    if len(sys.argv)!=2:raise ValueError('usage: pca_fixture.py NEW_COMMON_SOURCE.pt')
    out=Path(sys.argv[1])
    if out.exists():raise ValueError('new PCA common input required')
    for path,expected in PINS.items():
        if sha(path)!=expected:raise ValueError('PCA source/reference changed: '+str(path))
    torch.set_num_threads(1)
    fields=torch.load(FEATURES,weights_only=True)
    with np.load(REFERENCE,allow_pickle=False) as archive:
        ref={key:archive[key].copy() for key in ('ts_mu','ts_sd','idx_tr','tsem_m','tsem_V',
                                                'route_V_hi','singular_values','pc_tr')}
    if fields['features'].shape!=(2278,44239) or not np.array_equal(fields['rows'].numpy(),ref['idx_tr']):
        raise ValueError('source PCA rows/features differ')
    z=(fields['features'].numpy()-ref['ts_mu'])/ref['ts_sd']
    if z.dtype!=np.float64 or not np.isfinite(z).all():raise ValueError('invalid common standardized source')
    basis=np.concatenate((ref['tsem_V'],ref['route_V_hi']),axis=1)
    save_tensors(out,dict(source_Z=z,reference_mean=ref['tsem_m'],reference_basis=basis,
        reference_singular=ref['singular_values'],reference_pc=ref['pc_tr']))
    proof=dict(status='COMMON_SOURCE_PCA_INPUT_NO_FIT',source_rows=2278,width=44239,
        original_source_features_sha256=PINS[FEATURES],original_basis_sha256=PINS[REFERENCE],
        output_sha256=sha(out),producer_sha256=sha(__file__),query_labels_opened=False,
        accuracy_computed=False,Python_head_fit=False,PCA_fit=False)
    out.with_suffix('.json').write_text(json.dumps(proof,indent=2))


if __name__=='__main__':main()
