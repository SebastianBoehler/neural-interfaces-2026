"""Fail-closed smoke-to-remaining15 release; no query gold or scientific selection."""
import json
import subprocess
import sys
from pathlib import Path

from returning_assets import ROOT,sha

FULL=ROOT/'ni26_js321_20261117_returning_pair_cpu1'
COLLECT=ROOT/'ni26_js321_20261117_returning_pair_cpu_complete1'
GUARD=ROOT/'ni26_js321_20261117_returning_pair_guard2/output/pair_guard.json'


def submit(args,cwd):
    result=subprocess.check_output(['sbatch','--parsable',*args],cwd=cwd,text=True).strip()
    if not result.split(';')[0].isdigit():raise ValueError('unrecognized actual scheduler receipt')
    return result.split(';')[0]


def main(pilot):
    p=json.loads((FULL/'cell_0/proof.json').read_text())
    if (p['status'],p['index'],p['fold'],p['order'])!=('complete_actual_conditional_pair_cell_NOT_scored',0,'P04_s23','within0'):raise ValueError('complete canonical real-cell smoke required')
    if p['executor']!='cleanSIF_cpu_deterministic_validation_matched_independent_RNG':raise ValueError('explicit all-CPU smoke required')
    if p['accuracy_computed'] or p['query_labels_read'] or p['source_sha256']!=sha(FULL/'source.sha256') or p['guard_sha256']!=sha(GUARD) or p['predictions_sha256']!=sha(FULL/'cell_0/predictions.npz'):raise ValueError('smoke source/guard/asset scope failed')
    if len(p['records'])!=2 or sum(r['windows'] for r in p['records'])!=377 or p['official_cell_keys']!=json.loads((FULL/'input_preflight.json').read_text())['cases'][0]['official_cell_keys']:raise ValueError('smoke actual wholefold coverage failed')
    for r in p['records']:
        if not r['off_base_final_bitwise'] or not r['common_inner_gate_FT_bitwise'] or not r['source_immutable'] or max(r['owning_mix_maxabs'],r['raw_view_maxabs'])>1e-8 or any(v['ft_over'] for v in r['activation'].values()):raise ValueError('smoke active frame/runtime guard failed')
    accounting=subprocess.check_output(['sacct','-j',pilot,'-n','-P','-o','JobID,State,ElapsedRaw,MaxRSS'],text=True)
    batch=[l.split('|') for l in accounting.splitlines() if '.batch|' in l]
    if len(batch)!=1 or batch[0][1]!='COMPLETED' or not batch[0][3]:raise ValueError('actual completed smoke resource evidence required')
    rss=batch[0][3];units={'K':1.,'M':1024.,'G':1024.**2,'T':1024.**3};peak=float(rss[:-1])*units[rss[-1]]
    elapsed=int(batch[0][2])
    if peak>16*1024**2 or elapsed>6000:raise ValueError('smoke exceeded registered wholecell resource bound')
    states=subprocess.check_output(['squeue','--me','-h','-r','-o','%T'],text=True).splitlines();running=states.count('RUNNING');active=len(states)
    throttle=8 if peak<=8*1024**2 and running+8<=32 else 4
    if running+throttle>32 or active+16>244:raise ValueError('NO_RELEASE capacity32running/244active admission guard')
    subprocess.run(['sha256sum','-c','source.sha256'],cwd=FULL,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['sha256sum','-c','source.sha256'],cwd=COLLECT,check=True,stdout=subprocess.DEVNULL)
    array=submit([f'--array=1-15%{throttle}','--output=slurm-%A_%a.out','returning_pair_cpu.sbatch'],FULL)
    receipt=dict(status='actual_smoke_passed_remaining15_released_NOT_scored',pilot_job=pilot,pilot_proof_sha256=sha(FULL/'cell_0/proof.json'),
                 actual_smoke_elapsed_s=elapsed,actual_smoke_maxrss_KiB=peak,throttle=throttle,remaining_array=array,
                 owning_full_source_sha256=sha(FULL/'source.sha256'),controller_source_sha256=sha('source.sha256'),
                 user_running_at_release=running,user_active_at_release=active,accuracy_computed=False,scope='CONDITIONAL_ENGINEERING_ONLY')
    Path('release.json').write_text(json.dumps(receipt,indent=2))
    collector=submit([f'--dependency=afterok:{array}','--output=slurm-%j.out','returning_pair_collect.sbatch'],COLLECT)
    receipt['owning_collector']=collector;Path('release.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)


if __name__=='__main__':main(sys.argv[1])
