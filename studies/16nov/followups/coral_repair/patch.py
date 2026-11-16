"""Replace the sole CORAL call in a private SessionNet function, fail closed."""
import inspect
import textwrap
from loss import training_penalty


def install(session_module):
    original = session_module.SessionNet._train
    source = textwrap.dedent(inspect.getsource(original))
    anchor = 'dg.coral(tap.feat, self._grp[jd])'
    if source.count(anchor) != 1:
        raise ValueError('unexpected SessionNet CORAL implementation')
    source = source.replace(anchor, 'repair_penalty(tap.feat, self._grp[jd], yb, '
                            'self.repair_stats, dg.coral, self.repair_mode)', 1)
    namespace = dict(original.__globals__, repair_penalty=training_penalty)
    exec(compile(source, inspect.getsourcefile(original), 'exec'), namespace)
    session_module.SessionNet._train = namespace['_train']
    return original
