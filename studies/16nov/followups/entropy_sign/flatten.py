"""Change only the per-window entropy sign in the fixed §307 update."""
import importlib.util
from pathlib import Path

path=Path(__file__).resolve().parent.parent/'meta_adaptation/adapt.py'
source=path.read_text()
old='return entropy - .1 * diversity + .1 *'
new='return -entropy - .1 * diversity + .1 *'
if source.count(old)!=1: raise ValueError('unexpected entropy objective')
namespace={'__name__':'s310_flatten','__file__':str(path)}
exec(compile(source.replace(old,new,1),str(path),'exec'),namespace)
Session=namespace['Session']
loss=namespace['loss']
