"""Complete CPU functionality only after actual standard-probe allocation timeout."""
import json
import subprocess
import sys
from pathlib import Path


def main(variant,job):
    expected={'fir':'3117587','reuse':'3117585'}
    if expected.get(variant)!=job:raise ValueError('unregistered timeout parent')
    state=subprocess.check_output(['sacct','-j',job,'-X','-n','-o','State'],text=True).strip()
    if state!='TIMEOUT':
        Path('timeout_'+variant+'.json').write_text(json.dumps(dict(status='not_timeout_no_release',state=state)))
        return
    rows=subprocess.check_output(['squeue','-u','gfh527','-h','-o','%T'],text=True).splitlines()
    if sum(x=='RUNNING' for x in rows)>=32 or len(rows)>=244:raise ValueError('capacity guard')
    target=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_functional1')
    subprocess.run(['sha256sum','-c','source.sha256'],cwd=target,check=True,stdout=subprocess.DEVNULL)
    released=subprocess.check_output(['sbatch','--parsable','functional_'+variant+'_cpu.sbatch'],cwd=target,text=True).strip()
    Path('timeout_'+variant+'.json').write_text(json.dumps(dict(status='timeout_preserved_functional_only_released',
                                                              parent=job,job=released),indent=2))


if __name__=='__main__':main(*sys.argv[1:])
