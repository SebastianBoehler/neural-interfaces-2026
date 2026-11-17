import json
import sys
from pathlib import Path
import torch
STUDY=Path('/mnt/lustre/work/gehler/gfh527/ni26_factor354_v1_20261117')
sys.path.insert(0,str(STUDY))
from data354 import original
from gradient354 import check
torch.set_num_threads(2)
data,_,_,people=original.load()
result=check(data,people,'P02',STUDY/'output/P02')
Path('result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
