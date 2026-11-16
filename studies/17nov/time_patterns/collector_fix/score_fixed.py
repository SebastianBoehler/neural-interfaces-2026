"""Correct only §314 report formatting while retaining its validated scorer."""
import os
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent


def patch(source):
 anchor=' def contrast(a,b):'
 tail=" target=ROOT/'evidence/next_wave_16nov/s314_results.json'"
 if source.count(anchor)!=1 or source.count(tail)!=1:raise ValueError('unexpected original scoring report source')
 start=source.index(anchor);end=source.index(tail,start)
 replacement=''' from report import build_report
 report=build_report(official,pp,ORDERS,PEOPLE,folds,sha(ROOT/'s314-source.sha256'),summarize)
 report['collector_source_sha256']=sha(Path(os.environ['S314_COLLECTOR_MANIFEST']))
'''
 return source[:start]+replacement+source[end:]


def main():
 root=Path(os.environ['NI26_WS']).resolve(strict=True)
 path=root/'studies/17nov/time_patterns/score.py'
 sys.path.insert(0,str(HERE))
 exec(compile(patch(path.read_text()),str(path),'exec'),{'__name__':'__main__','__file__':str(path)})


if __name__=='__main__':main()
