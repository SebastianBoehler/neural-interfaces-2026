"""Release fixed finalist warmed timing only after complete actual functional proof."""
import json
import subprocess
from pathlib import Path


def main():
    root=Path('/mnt/lustre/work/gehler/gfh527')
    proof=root/'ni26_js321_20261117_functional1/functional_reuse_cuda/functional_cuda.json'
    report=json.loads(proof.read_text())
    if report['status']!='complete_functional_only_matrix_passed_NOT_timing_qualified' or report['variant']!='reuse':
        raise ValueError('full fixed-finalist functional proof absent')
    if len(report['rows'])!=6:raise ValueError('incomplete functional rows')
    rows=subprocess.check_output(['squeue','-u','gfh527','-h','-o','%T'],text=True).splitlines()
    if sum(x=='RUNNING' for x in rows)>=32 or len(rows)>=244:raise ValueError('capacity guard')
    target=root/'ni26_js321_20261117_warm1'
    subprocess.run(['sha256sum','-c','source.sha256'],cwd=target,check=True,stdout=subprocess.DEVNULL)
    job=subprocess.check_output(['sbatch','--parsable','warm_timing.sbatch'],cwd=target,text=True).strip()
    Path('warm_release.json').write_text(json.dumps(dict(status='fixed_finalist_warm_timing_released',job=job,
                                                       functional_proof=str(proof),old_failures_preserved=True),indent=2))


if __name__=='__main__':main()
