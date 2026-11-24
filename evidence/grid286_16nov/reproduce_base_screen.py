from pathlib import Path
p=Path('<repo>/studies/15nov/grid283/score_completed.py')
s=p.read_text()
orig="new=('fb_dist1','fb_dist2','fb_dist4','eegn_dist1','eegn_dist2','eegn_dist4','fbl_coral10_am50','fbw2_dist2','tsm','tsm_lr3e4','tsm_lr3e3')"
old=('fb_dist1','fb_dist2','fb_dist4','eegn_dist1','eegn_dist2','eegn_dist4','fbw2_dist2','tsm','tsm_lr3e4','tsm_lr3e3')
extra=tuple(f'{f}_coral{w}' for f in ('fb','fbl','sinc','eegn') for w in (1,3,10,30,100) if not (f=='fb' and w==1))+tuple(f'{f}_coral10_am50' for f in ('fb','fbl','sinc'))
assert orig in s;s=s.replace(orig,'new='+repr(extra+old))
oldroot="base=Path(C.D) if f in old else root/('tsm' if f.startswith('tsm') else 'grid')"
assert oldroot in s
s=s.replace(oldroot,"base=Path(C.D) if f in old else (repo/'runs/grid286_287_recovery/grid' if f in "+repr(extra)+" else root/('tsm' if f.startswith('tsm') else 'grid'))")
oldfam="fam='fb' if f.startswith('fb_') or f.startswith('fbw') else 'eegn' if f.startswith('eegn_') else 'fbl' if f.startswith('fbl_') else 'tsm'"
assert oldfam in s
s=s.replace(oldfam,"fam='fb' if f.startswith('fbw') else f.split('_')[0]")
s=s[:s.index("print('Paired distillation effect")]
s += '''\nprint("Relative to adding plain EEGNet (same seed weight)")\ne0=ba(base6+nets(["eegn"],3))\nfor f in ("eegn_coral1","eegn_coral3","eegn_coral10","eegn_dist2"):\n print(f,stat(ba(base6+nets([f],3))-e0))\nprint("Rank across available complete 32 arms by add-six; provisional until B1 and fb coral1 are incorporated")\nfor i,f in enumerate(sorted(new,key=lambda f:ba(base6+nets([f],3)).mean(),reverse=True),1):\n v=ba(base6+nets([f],3))-b6\n print(i,f,stat(v))\n'''
exec(compile(s,str(p),'exec'),{'__file__':str(p),'__name__':'__main__'})
