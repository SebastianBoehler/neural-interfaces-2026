"""Time unchanged existing full-chain cell; no metrics or stage changes."""
import json
import os
import sys
import time
from pathlib import Path
from cell import main
root=Path(os.environ['NI26_WS']);arm,order=sys.argv[1:];start=time.monotonic()
main(arm,order)
(root/f'timing_{arm}_{order}.json').write_text(json.dumps(dict(arm=arm,order=order,seconds=time.monotonic()-start,cpus=6)))
