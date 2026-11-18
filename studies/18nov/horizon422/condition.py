"""Release the order-confirmation jobs only after the preregistered effect."""
import json
from pathlib import Path

r = json.loads(Path('../ni26_causal422_20261118/result.json').read_text())['effects']['optimizer']
proceed = r['mean'] >= 1.5 and r['ci80'][0] > 0
Path('condition.json').write_text(json.dumps(dict(proceed=proceed, mean=r['mean'], ci80=r['ci80']))+'\n')
raise SystemExit(0 if proceed else 42)
