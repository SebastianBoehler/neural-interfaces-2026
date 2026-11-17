import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('patch318',HERE/'patch.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class PatchTests(unittest.TestCase):
    def test_factor_and_member_weights_on_actual_patched_mix(self):
        source=Path(os.environ['S318_SOURCE']).read_text()
        start=source.index('FACTOR = '); end=source.index('_savez = ',start)
        block=source[start:end]
        class S: N={'other':2}
        class A:
            EXTRA={}
            @staticmethod
            def mix(bn,person,fams,extra=()):
                values={'sc_fb':np.array([[.7,.2,.1],[.1,.7,.2]]),
                        'sc_fbl':np.array([[.1,.7,.2],[.6,.1,.3]]),
                        'sc_sinc':np.array([[.3,.4,.3],[.3,.2,.5]]),
                        'sc_fbraw':np.array([[.2,.3,.5],[.5,.3,.2]])}
                values.update({k:v['person'] for k,v in A.EXTRA.items()})
                total=sum(w for _,w in extra)
                return sum(values[n]*w for n,w in extra)/total, 'fixed'
        original = A.mix
        extra=[('sc_fb',1.),('sc_fbl',2.),('sc_sinc',3.),('sc_fbraw',4.)]
        base=A.mix('b','person',None,extra)[0]
        for name,add,swap,zero,factor in [('zero318',['fb_cc30'],[],True,1.),('FBL2',[],[],False,2.),
                                          ('SA',['fb_cc30'],[],False,1.),('SS',[],['fb_cc30'],False,1.),
                                          ('JA',['fb_cc30'],[],False,2.),('JS',[],['fb_cc30'],False,2.)]:
            A.EXTRA={}
            # Restore the method that each generated wrapper captures.
            A.mix=staticmethod(original)
            scope=dict(NAME=name,FAMS_=add+swap,ADD=add,SW=swap,ZERO=zero,np=np,A=A,S=S,p0='person',OTHER='other',
                       CUR={'P':np.array([1,0])},N=2,LOG=[],stream=lambda f,b:np.array([[.8,.1,.1],[.1,.2,.7]]))
            exec(block,scope)
            actual=A.mix('b','person',None,extra)[0]
            e=[(n,w*factor if n=='sc_fbl' else w) for n,w in extra if not (swap and n=='sc_fb')]
            if add or swap:e.append(('nm_fb_cc30',0. if zero else (1. if swap else 2.5)))
            expected=original('b','person',None,e)[0]
            np.testing.assert_allclose(actual,expected,rtol=0,atol=0)
            self.assertTrue(np.array_equal(actual,base) if zero else not np.array_equal(actual,base))

if __name__=='__main__': unittest.main()
