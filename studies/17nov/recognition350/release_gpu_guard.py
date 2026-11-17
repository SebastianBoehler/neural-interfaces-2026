"""After actual smoke, release only the preregistered remaining15; never inspect quality."""
import json
import subprocess
from pathlib import Path

from returning_assets import ROOT,sha

CELLS=ROOT/'ni26_recognition350_20261117_whole2'
GUARD=ROOT/'ni26_recognition350_20261117_gpu_guard2'


def main():
    p=json.loads((CELLS/'cell_0/proof.json').read_text());g=json.loads((GUARD/'output/guard.json').read_text())
    if p['status']!='POLICY350_COMPLETE_WHOLE_CELL_NOT_SCORED' or p['index']!=0 or p['source_sha256']!=sha(CELLS/'source.sha256') or p['predictions_sha256']!=sha(CELLS/'cell_0/predictions.npz') or p['owning_guard_sha256']!=sha(GUARD/'output/guard.json') or g['status']!='POLICY350_GPU_SOURCE_FUNCTIONAL_PASS_NOT_ACCURACY':raise ValueError('actual immutable source/smoke proof missing')
    if p['seconds']>480 or len(p['records'])!=2 or p['query_labels_loaded']:raise ValueError('whole smoke cost/coverage guard failed')
    # Spent CPU180.2min + A100CPU30 +16cells256 +release2+collector4 =472.2/60=7.870h.
    if 180.2+2*15+16*2*8+2+4>474:raise ValueError('eightCPUh preregistered hard budget exceeded')
    running=subprocess.check_output(['squeue','--me','-h','-r','-t','RUNNING'],text=True).splitlines();active=subprocess.check_output(['squeue','--me','-h','-r'],text=True).splitlines()
    if len(running)+8>32 or len(active)+16>244:raise ValueError('user-wide capacity guard prevents release')
    job=subprocess.check_output(['sbatch','--parsable','--array=1-15%8','--output=cell-%A_%a.log','cell_gpu_guard.sbatch'],cwd=CELLS,text=True).strip()
    collect=subprocess.check_output(['sbatch','--parsable','--dependency=afterok:'+job,'--output=collector-%j.log','collect_gpu_guard.sbatch'],text=True).strip()
    Path('release.json').write_text(json.dumps(dict(status='POLICY350_REMAINING15_RELEASED_AFTER_ACTUAL_SMOKE',array_job=job,collector_job=collect,pilot_proof_sha256=sha(CELLS/'cell_0/proof.json'),guard_sha256=sha(GUARD/'output/guard.json'),smoke_seconds=p['seconds'],maximum_allocated_CPUhours=7.87,accuracy_read=False),indent=2))
    print(job,collect,flush=True)


if __name__=='__main__':main()
