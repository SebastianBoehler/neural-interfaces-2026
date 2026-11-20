"""Lane R4 submit driver (runs on the Mac, only calls ssh + sbatch/squeue): Galvani allows 300 queued jobs per user and other lanes share them,
so units (returning array of an arm; NI-V chain of an arm) are submitted one at a time when they fit. usage: driver.py UNIT... ; unit = ret:ARM | niv:ARM"""
import subprocess
import sys
import time

HOST, LIMIT = "mlcloud-galvani", 285
RET = "/mnt/lustre/work/gehler/gfh527/ni26_laneR4_20261120/ret"
NIV = "/mnt/lustre/work/gehler/gfh527/ni26_niv_v1_20261119"


def ssh(cmd):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, cmd], capture_output=True, text=True)
    return r.returncode, "\n".join(l for l in r.stdout.splitlines() if "pq.html" not in l).strip(), r.stderr


def queued():
    rc, out, _ = ssh("squeue -u gfh527 -h -r | wc -l")
    return int(out) if rc == 0 and out.isdigit() else 10 ** 6


def script(unit):
    kind, arm = unit.split(":")
    if kind == "ret":
        n = 4 if arm == "R0z" else 16
        arr = "0,1,8,9" if arm == "R0z" else "0-15"
        return n, f"cd {RET} && echo ret_{arm} $(RET_VAR={arm} sbatch --parsable --export=ALL --array={arr} ret_worker.sbatch)"
    t = f"hardcbB_{arm}"
    return 22, f"""set -e; cd {NIV}
X="--exclude=galvani-cn122,galvani-cn116"
cp1=$(sbatch --parsable --partition=cpu-galvani --mem=2G --time=00:10:00 -J ni26-R4-cp -o logs/R4cp-%j.out --wrap="set -e; cd {NIV}; cp -n {RET}/zips/{t}.zip zips/{t}.zip; sha256sum zips/{t}.zip >> inputs.sha256")
n=$(sbatch --parsable $X --dependency=afterok:$cp1 --array=0-8 {RET}/../laneR4/niv_score_pack.sbatch {t}.zip {t})
e=$(sbatch --parsable --dependency=afterok:$cp1 enrol.sbatch {t}.zip {t})
r=$(sbatch --parsable $X --dependency=afterok:$e --array=0-8 enrol_run.sbatch {t} ret)
c1=$(sbatch --parsable --dependency=afterok:$n:$r collect.sbatch {t} hardcbB)
c2=$(sbatch --parsable --dependency=afterok:$n:$r collect.sbatch {t} cap250)
echo niv_{arm} cp $cp1 bnew $n enrol $e ret $r collectors $c1 $c2"""


for unit in sys.argv[1:]:
    n, cmd = script(unit)
    while True:
        if queued() + n <= LIMIT:
            rc, out, err = ssh(cmd)
            print(time.strftime("%H:%M:%S"), unit, rc, out, err.strip()[-200:] if rc else "", flush=True)
            if rc == 0:
                break
        time.sleep(240)
