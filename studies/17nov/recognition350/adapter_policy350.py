"""One causal tail; select an outer recipe using only the existing recognition latch."""
from .adapter_inputreuse import ReuseEnsemble
from .adapter_shared import EnsembleJS
from .v6chain import Ensemble6


class RecognitionEnsemble(ReuseEnsemble):
    def __init__(self,z,dev,root):
        mode=int(z['recognition_policy350'])
        if mode not in (0,1,2):
            raise ValueError('registered off/all-on/runtime-unrecognized mode required')
        self.policy350=mode
        if bool(int(z['js_on']))!=(mode!=0):
            raise ValueError('policy mode and active auxiliary bank disagree')
        super().__init__(z,dev,root)
        self.base_a=float(self.w[0]*(1-self.w_rest))
        self.reset_state()

    def reset_state(self):
        super().reset_state()
        self.policy350_js=0
        self.policy350_base=0
        if hasattr(self,'base_a'):
            self.routeB.a=self.base_a/(1+self.js_unit) if self.js_on else self.base_a

    def mix6(self,mix,members,p_ft):
        if not self.js_on:
            self.policy350_base+=1
            return Ensemble6.mix6(self,mix,members,p_ft)
        self.consume_current_input()
        use_js=self.policy350==1 or not self.recB
        self.routeB.a=self.base_a/(1+self.js_unit) if use_js else self.base_a
        if use_js:
            self.policy350_js+=1
            return EnsembleJS.mix6(self,mix,members,p_ft)
        self.policy350_base+=1
        return Ensemble6.mix6(self,mix,members,p_ft)
