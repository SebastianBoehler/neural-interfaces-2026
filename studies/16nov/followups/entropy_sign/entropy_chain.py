"""Inject the registered §310 target deltas with the original chain wrapper."""
import os
import sys
from pathlib import Path

ARMS=('Rzero','Anchor','Sharpen','Flatten')
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')


def main():
    root=Path(os.environ['NI26_WS']).resolve(strict=True)
    path=root/'studies/16nov/followups/native_core/member_chain.py'
    source=path.read_text()
    replacements={"ARMS = ('Rzero', 'Nfixed', 'Ntrain')":f'ARMS = {ARMS!r}',
                  '            self.track(Path(__file__).resolve())':
                  '            self.track(Path(__file__).resolve())\n            self.track('+repr(str(Path(__file__).resolve()))+')'}
    for old,new in replacements.items():
        if source.count(old)!=1: raise ValueError(f'unexpected member wrapper: {old}')
        source=source.replace(old,new,1)
    exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path),'os':os,'sys':sys})


if __name__=='__main__':main()
