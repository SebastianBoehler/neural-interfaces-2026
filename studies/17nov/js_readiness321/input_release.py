"""Reuse guarded worker release only after explicit readonly-current-input checks."""
import json
from pathlib import Path

import shared_release


def main():
    shared_release.SHARED=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_inputreuse2')
    r=json.loads((shared_release.SHARED/'optimization_guard.json').read_text())
    if r['execution_variant']!='sharedFIR_readonly_current_original_input' or not all(c['input_reuse_checked'] for c in r['cases'].values()):
        raise ValueError('readonly current-step callback guard missing')
    shared_release.main(retry_script='input_release.sbatch')


if __name__=='__main__':main()
