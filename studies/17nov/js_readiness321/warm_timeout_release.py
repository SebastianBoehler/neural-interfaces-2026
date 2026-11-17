"""Release complete identical warm protocol only after actual allocation timeout."""
import json
import subprocess
from pathlib import Path


def main():
    state=subprocess.check_output(['sacct','-j','3117650','-X','-n','-o','State'],text=True).strip()
    if state!='TIMEOUT':
        Path('decision.json').write_text(json.dumps(dict(status='no_timeout_no_release',state=state)))
        return
    rows=subprocess.check_output(['squeue','-u','gfh527','-h','-o','%T'],text=True).splitlines()
    if sum(x=='RUNNING' for x in rows)>=32 or len(rows)>=244:raise ValueError('capacity guard')
    target=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_warm3')
    subprocess.run(['sha256sum','-c','source.sha256'],cwd=target,check=True,stdout=subprocess.DEVNULL)
    job=subprocess.check_output(['sbatch','--parsable','warm_timing.sbatch'],cwd=target,text=True).strip()
    Path('decision.json').write_text(json.dumps(dict(status='old_timeout_preserved_complete_protocol_released',job=job,
                                                   partial_rows_reused=False),indent=2))


if __name__=='__main__':main()
