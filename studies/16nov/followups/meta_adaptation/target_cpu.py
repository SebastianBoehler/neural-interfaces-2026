"""Explicit CPU route for the unchanged §307 held-person producer."""
from pathlib import Path

path = Path(__file__).with_name('target.py')
source = path.read_text()
for old,new in (("from bridge import Native", "from cpu_crossfit import CPUReference as Native"),
                ("device = torch.device('cuda:0')", "device = torch.device('cpu')")):
    if source.count(old) != 1: raise ValueError(f'unexpected target route anchor: {old}')
    source = source.replace(old,new,1)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
