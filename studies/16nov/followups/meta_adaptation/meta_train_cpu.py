"""Explicit CPU route for the identical §307 source meta-training code."""
from pathlib import Path

path = Path(__file__).with_name('meta_train.py')
source = path.read_text()
old = "device = torch.device('cuda:0')"
if source.count(old) != 1: raise ValueError('unexpected meta-training device declaration')
source = source.replace(old,"device = torch.device('cpu')",1)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
