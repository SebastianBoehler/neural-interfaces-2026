"""Own complete CPU functionality status; no labels, accuracy or automatic retries."""
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root=Path('/mnt/lustre/work/gehler/gfh527/ni26_js321_20261117_functional1')
    subprocess.run(['sha256sum','-c','source.sha256'],cwd=root,check=True,stdout=subprocess.DEVNULL)
    rows={}
    for variant,job in (('fir','3117757'),('reuse','3117756')):
        state=subprocess.check_output(['sacct','-j',job,'-X','-n','-o','State'],text=True).strip()
        if state!='COMPLETED':
            rows[variant]=dict(job=job,status='incomplete_preserve_no_retry',slurm_state=state)
            continue
        path=root/('functional_'+variant+'_cpu')/'functional_cpu.json'
        report=json.loads(path.read_text())
        if report['variant']!=variant or report['device']!='cpu' or report['status']!='complete_functional_only_matrix_passed_NOT_timing_qualified':
            raise ValueError('owning full functionality proof absent')
        expected={(m,n) for m in ('known','unrecognized') for n in (190,378,800)}
        if {(r['mode'],r['windows']) for r in report['rows']}!=expected or len(report['rows'])!=6:
            raise ValueError('incomplete CPU functional matrix')
        for row in report['rows']:
            for arm in ('base','js'):
                a=row[arm]['activity']
                if arm=='js' and row[arm]['js_calls']!=row['windows']:raise ValueError('auxiliary inactive')
                if row['mode']=='known' and (row[arm]['recognized_at']!=10 or a['eligible'] or a['ft_refits'] or a['cp_fused']):
                    raise ValueError('known first-recognition path differs')
                if row['mode']=='unrecognized' and row[arm]['recognized_at']>=0:raise ValueError('unknown path recognized')
                if row['mode']=='unrecognized' and (not a['eligible'] or a['ft_over'] or
                    a['ft_refits']!=a['expected_ft'] or a['cp_refits']!=a['expected_cp'] or
                    a['cp_fused']!=a['expected_fused']):raise ValueError('late CPU stage activity differs')
        rows[variant]=dict(job=job,status='complete_functionality_passed_NOT_timing_qualified',
                           report=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    Path('cpu_complete.json').write_text(json.dumps(dict(status='owning_CPU_status_complete',variants=rows,
                                                        old_failures_preserved=True,accuracy_computed=False,
                                                        automatic_retry=False),indent=2))


if __name__=='__main__':main()
