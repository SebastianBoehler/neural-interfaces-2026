"""Generate exactly one isolated validation change; original library/source stay untouched."""
import hashlib
import json
from pathlib import Path

root=Path('/mnt/lustre/work/gehler/gfh527/ni26_s311_cpu_kernel_v1_20261116')
manifest=root/'source.sha256'
if hashlib.sha256(manifest.read_bytes()).hexdigest()!='fb09442bf275646faf14bdf8da37d48c3226adb1ae141ad8fc5cda77ab7d3312':raise ValueError('originalsourcepin')
pins={name:pin for pin,name in (line.split(maxsplit=1) for line in manifest.read_text().splitlines())}
name='studies/16nov/followups/native_core/head.cpp';original=(root/name).read_text()
if hashlib.sha256(original.encode()).hexdigest()!=pins[name]:raise ValueError('originalheadchanged')
old='require(torch::isfinite(result).all().item<bool>(), "nonfinite matrix");'
new='for (std::int64_t index = 0; index < n * d; ++index) require(std::isfinite(x[index]), "nonfinite matrix");'
if original.count(old)!=1:raise ValueError('validationanchorchanged')
changed=original.replace(old,new);out=Path('build/head_validation.cpp');out.write_text(changed)
Path('build/change.json').write_text(json.dumps(dict(original_sha256=pins[name],changed_sha256=hashlib.sha256(changed.encode()).hexdigest(),old=old,new=new),indent=2))
