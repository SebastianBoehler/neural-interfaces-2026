"""Run the unchanged §304 job body inside its private recovery snapshot."""
import os
import subprocess
import sys
from pathlib import Path
root=Path(os.environ['NI26_WS']).resolve(strict=True)
path=root/'studies/16nov/followups/coral_repair/chain/job.sbatch'
source=path.read_text();old='cd /mnt/lustre/work/gehler/gfh527/ni26_s304_coral_chain_20261116'
if source.count(old)!=1:raise ValueError('unexpected immutable §304 job body')
source=source.replace(old,'cd "$NI26_WS"',1)
subprocess.run(['bash','-s','--',*sys.argv[1:]],input=source,text=True,check=True)
