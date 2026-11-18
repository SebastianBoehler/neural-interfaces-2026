"""Fetch a predeclared small range of the version-pinned open dataset."""
import configparser
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote
from urllib.request import Request,urlopen

import numpy as np

TAG='https://raw.githubusercontent.com/OpenNeuroDatasets/ds004148/1.0.1/'
STORE='https://s3.amazonaws.com/openneuro.org/ds004148/'
TASKS=('mathematic','memory','music')
TIMES=(30,150,270)


def sha(b):return hashlib.sha256(b).hexdigest()


def get(url,headers=None):
    with urlopen(Request(url,headers=headers or {}),timeout=25) as r:
        return r.read(),r.status,dict((k.lower(),v) for k,v in r.headers.items())


def current_file(key):
    ptr=get(TAG+key)[0].decode()
    match=re.search(r'MD5E-s(\d+)--([a-f0-9]{32})',ptr)
    assert match is not None
    with urlopen(Request(STORE+key,method='HEAD'),timeout=20) as r:
        assert int(r.headers['Content-Length'])==int(match[1])
        assert r.headers['ETag'].strip('"')==match[2]
        version=r.headers['x-amz-version-id']
    return STORE+key+'?versionId='+quote(version,safe=''),int(match[1]),match[2],version


def main(person,smoke):
    rosters=list(csv.DictReader(io.StringIO(get(TAG+'participants.tsv')[0].decode()),delimiter='\t'))
    assert json.loads(get(TAG+'dataset_description.json')[0])['License']=='CC0'
    subject=f'sub-{person:02d}'
    roster=next(r for r in rosters if r['participant_id']==subject)
    out=[];proof=[];order=[]
    for s in ((3,) if smoke else (1,2,3)):
        ranks=[int(roster[n+'_order_session'+str(s)]) for n in ('Subtraction','Memory','Music')]
        assert sorted(ranks)==[3,4,5] and ranks[0]==6-s
        order.append(ranks);day=[]
        for task in TASKS[:1] if smoke else TASKS:
            stem=f'{subject}/ses-session{s}/eeg/{subject}_ses-session{s}_task-{task}_'
            raw,size,md5,version=current_file(stem+'eeg.eeg')
            hdr,_,_,_=current_file(stem+'eeg.vhdr')
            text=get(hdr)[0].decode('utf-8-sig')
            cfg=configparser.ConfigParser(interpolation=None)
            cfg.read_string(text[text.index('[Common Infos]'):])
            ci=cfg['Common Infos']; n=int(ci['NumberOfChannels'])
            assert ci['DataOrientation']=='MULTIPLEXED' and float(ci['SamplingInterval'])==2000
            assert cfg['Binary Infos']['BinaryFormat']=='IEEE_FLOAT_32'
            meta=json.loads(get(TAG+stem+'eeg.json')[0])
            assert meta['SamplingFrequency']==500 and meta['RecordingDuration']==300 and meta['EEGReference']=='FCz'
            assert size==n*500*300*4
            channels=[cfg['Channel Infos'][f'ch{i}'].split(',') for i in range(1,n+1)]
            assert all(float(c[2])==1 for c in channels)
            names=[c[0] for c in channels]
            tab=list(csv.DictReader(io.StringIO(get(TAG+stem+'channels.tsv')[0].decode()),delimiter='\t'))
            bids=[r['name'] for r in tab]
            assert len(set(bids))==len(bids) and all(r['units'] in ('uV','µV') for r in tab)
            assert all(a==b or (a,b)==('Cpz','CPz') for a,b in zip(names,bids,strict=True))
            names=bids
            events=list(csv.DictReader(io.StringIO(get(TAG+stem+'events.tsv')[0].decode()),delimiter='\t'))
            frames=[]
            for t in TIMES[:1] if smoke else TIMES:
                assert not any(t<=float(r['onset'])<t+4 for r in events if r['type']!='SyncStatus')
                a=t*500*n*4;b=(t+4)*500*n*4-1
                cache=Path(f'ranges/{md5}_{a}_{b}.bin')
                cache.parent.mkdir(exist_ok=True)
                if cache.exists():
                    payload=cache.read_bytes()
                    assert sha(payload)==cache.with_suffix('.sha256').read_text()
                else:
                    payload,status,h=get(raw,{'Range':f'bytes={a}-{b}'})
                    assert status==206 and h['content-range']==f'bytes {a}-{b}/{size}' and len(payload)==b-a+1
                    assert h['etag'].strip('"')==md5
                    cache.with_suffix('.sha256').write_text(sha(payload));cache.write_bytes(payload)
                wave=np.frombuffer(payload,dtype='<f4').reshape(2000,n).T
                assert np.isfinite(wave).all() and np.std(wave)>0
                frames.append(wave.copy())
                proof.append(dict(task=task,session=s,seconds=t,md5=md5,version=version,
                                  bytes=len(payload),range_sha256=sha(payload)))
            if not day and not out:canonical=names
            assert set(canonical)==set(names)
            day.append(np.stack(frames)[:,[names.index(k) for k in canonical]])
        out.append(day)
    if not smoke:
        np.savez(f'{subject}.npz',x=np.array(out),names=canonical,order=order)
    Path(('smoke' if smoke else subject)+'.json').write_text(json.dumps(proof))
if __name__=='__main__':main(int(sys.argv[1]),len(sys.argv)>2)
