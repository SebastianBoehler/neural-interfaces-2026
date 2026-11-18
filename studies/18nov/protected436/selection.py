"""Leave one external participant out of the protected-versus-cap250 choice."""
import json
from pathlib import Path

import numpy as np

result = json.loads(Path("result436.json").read_text())
delta = np.asarray(result["person_delta_pp"])
if len(delta) != 9 or result["arms"] != ["cap250", "protected"]:
    raise ValueError("expected complete two-arm external result")
source_means = (delta.sum() - delta) / (len(delta) - 1)
choose = source_means > 0
print(json.dumps(dict(people=result["people"], selected_protected=choose.tolist(),
                      person_held_delta_pp=float((choose*delta).mean()),
                      smallest_eight_person_fixed_delta_pp=float(source_means.min())), indent=2))
