"""Check the paired scoring implementation on real sealed predecessor predictions."""
import numpy as np
from score import ancestry,measure,summary
from study import DATA,PANELS,old,panel


def main():
    with np.load(DATA/'proteus_windows.npz') as z:
        p=np.array([v.split('/')[-1] for v in z['subject']])
        y,iface,session=z['y'],z['interface'],z['session']
    with np.load(old.source.INPUT/'inputs.npz') as z:
        orders=z['orders'].tolist()
    inputs=[]
    for i,hold in enumerate(panel.SOURCES):
        held=panel.DEV+(hold,)
        ix,outputs,_=ancestry.read(PANELS/str(i),PANELS/str(i),held,p,orders)
        inputs.append((held,ix,{('full','frozen'):outputs['full']['Z']}))
    weights=[len(set(session[p==a])) for a in panel.SOURCES+panel.DEV]
    a=measure(inputs,y,iface,session,p,False)['full','frozen']
    np.testing.assert_allclose(summary(a,weights)['official'][0],76.2793,atol=.0001,rtol=0)
    for change in [0.,1.]:
        s=summary((a+change)-a,weights)
        np.testing.assert_allclose(s['official'],change,atol=1e-12,rtol=0)
        np.testing.assert_allclose(s['gap'],0.,atol=1e-12,rtol=0)
    print('OK')


if __name__=='__main__':
    main()
