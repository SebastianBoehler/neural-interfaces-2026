"""Release330 using engineering/source-fit guards only, never329sourceaccuracy."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/lustre/work/gehler/gfh527/ni26_labram329_2080_v1_20261117')
SOURCE_SHA='057627db1f85aab13e0cab308f0a21d4e7c6645202b86fe2202160f4e4e44833'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for folder in (ROOT,SOURCE):
        for line in (folder/'source.sha256').read_text().splitlines():
            expected,name=line.split(maxsplit=1)
            if sha(folder/name)!=expected:raise ValueError('immutable sourcechanged')
    if sha(SOURCE/'source.sha256')!=SOURCE_SHA:raise ValueError('329sourceidentitydiffers')
    guard=json.loads((SOURCE/'compatibility.json').read_text());readout=json.loads((SOURCE/'fitted_readout_guard.json').read_text())
    if not guard['passed'] or guard['labels_opened'] or '2080 Ti' not in guard['device']:raise ValueError('passed2080engineeringrequired')
    if guard['peak_gpu_bytes']>8*1024**3 or guard['comparisons']['pretrained']['B1_median_ms']>=100:raise ValueError('measuredresourceboundfailed')
    if len(readout)!=9 or any(v['probability_max_abs']>1e-3 or v['nontied_argmax_changes'] for v in readout.values()):raise ValueError('actualfittedreadoutnumericalguardfailed')
    # Do not open source_result.json: release cannot depend on its queryaccuracies.
    states=subprocess.check_output(['squeue','-h','-u','gfh527','-o','%T']).decode().splitlines()
    if len(states)>=244 or sum(s in ('RUNNING','COMPLETING') for s in states)>=32:raise ValueError('accountcapacitynotavailable;norelease')
    submitted=subprocess.check_output(['sbatch','--parsable','train.sbatch'],cwd=ROOT).decode().strip()
    report=dict(released_job=submitted,source_sha256=sha(ROOT/'source.sha256'),prior_source_sha256=SOURCE_SHA,
        read329queryaccuracy=False,source_banks_sha256=sha(SOURCE/'source_banks.npz'),compatibility_sha256=sha(SOURCE/'compatibility.json'),readoutguard_sha256=sha(SOURCE/'fitted_readout_guard.json'))
    with (ROOT/'release.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
