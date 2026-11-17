"""Validation-only deterministic CUDA original self-control; production modes unchanged."""
import json
import os
from pathlib import Path

if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('workspace configuration must precede Python/CUDA initialization')
import torch
import ft_repeat


def main():
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    ft_repeat.main();p=Path('ft_repeat.json');r=json.loads(p.read_text())
    passed=max(r['max_errors'].values())<=1e-8 and r['label_changes']==0 and r['ft_refits']==[20,20] and r['cp_refits']==[10,10] and r['cp_fused']==[125,125]
    r.update(status='deterministic_original215_control_passed' if passed else 'deterministic_original215_control_failed',
             workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],production_executor_unchanged=True)
    p.write_text(json.dumps(r,indent=2))
    if not passed:raise ValueError('strict deterministic original self-control failed;no tolerance fitting')


if __name__=='__main__':main()
