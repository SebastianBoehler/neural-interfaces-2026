"""One pinned DUCC plus two isolated base TUs, retaining all runtime objects."""
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

root=Path("/mnt/lustre/work/gehler/gfh527")
old=root/"ni26_native360_package_v1_20261117"
actual=root/"ni26_native366_copy_v1_20261117/profile_package"
ducc=root/"ni26_native366_ducc_v3_20261117/libhilbert_probe.so"
profile=Path.cwd()/"profile_package"
shutil.copytree(actual,profile,copy_function=os.link,
                ignore=lambda d,n:{"libni26_base.so"}&set(n))
os.link(ducc, profile/ducc.name)
cfg=old/"build/CMakeFiles/ni26_base.dir"
settings=dict(line.split(" = ",1) for line in (cfg/"flags.make").read_text().splitlines() if " = " in line)
flags=sum((shlex.split(settings[k]) for k in ("CXX_DEFINES","CXX_INCLUDES","CXX_FLAGS")),[])
link=shlex.split((cfg/"link.txt").read_text())
for file in ("core","filter"):
    p=Path.cwd()/f"{file}.o"
    subprocess.run([link[0],"-I"+str(Path.cwd()/"include"),*flags,
                    "-c",str(p.with_suffix(".cpp")),"-o",str(p)],check=True)
    obj=f"CMakeFiles/ni26_base.dir/src/{file}.cpp.o"
    assert link.count(obj)==1
    link[link.index(obj)]=str(p)
out=profile/"libni26_base.so"
link[link.index("-o")+1]=str(out)
assert sum(v.startswith("-Wl,-rpath,") for v in link)==1
link=["-Wl,-rpath,$ORIGIN" if v.startswith("-Wl,-rpath,") else v for v in link]
subprocess.run([*link,"-L"+str(profile),"-lhilbert_probe"],cwd=old/"build",check=True)
dyn=lambda p: subprocess.check_output(["readelf","-d",p],text=True)
rpat=r"\((?:RPATH|RUNPATH)\).*\[(.*?)\]"
r=re.findall(rpat,dyn(out))
if r!=["$ORIGIN"]:
    assert r==["$ORIGIN:"+str(root/"flightrl-puffer5.pmbvox/toolchain/lib")]
    subprocess.run(["cmake",f"-DLIBRARY={out}",f"-DOLD_RPATH={r[0]}",
                    "-P",str(old/"normalize_rpath.cmake")],check=True)
assert re.findall(rpat,dyn(out))==["$ORIGIN"]
libs=lambda p:set(re.findall(r"\(NEEDED\).*\[(.*?)\]",dyn(p)))
assert libs(out)==libs(actual/"libni26_base.so")|{ducc.name}
Path("linked.elf").write_text(dyn(out))
print("NATIVE_CORE_DUCC_LINK_PASS",flush=True)

ea=root/"ni26_native366_ea_native_v1_20261117/libea_probe.so"
os.link(ea,profile/ea.name)
cfg=old/"build/CMakeFiles/ni26_runtime.dir"
settings=dict(line.split(" = ",1) for line in (cfg/"flags.make").read_text().splitlines() if " = " in line)
flags=sum((shlex.split(settings[k]) for k in ("CXX_DEFINES","CXX_INCLUDES","CXX_FLAGS")),[])
link=shlex.split((cfg/"link.txt").read_text())
localinput=Path.cwd()/"networks_input.o"
subprocess.run([link[0],"-I"+str(old/"native/src"),*flags,"-c",
                str(localinput.with_suffix(".cpp")),"-o",str(localinput)],check=True)
prior=root/"ni26_native366_copy_v1_20261117"
for name,path in {"finetune":prior/"finetune_probe.o","networks_extras":prior/"extras.o",
                  "networks_input":localinput}.items():
    obj=f"CMakeFiles/ni26_runtime.dir/src/{name}.cpp.o"
    assert link.count(obj)==1
    link[link.index(obj)]=str(path)
out=profile/"libni26_runtime_new.so"
link[link.index("-o")+1]=str(out)
assert sum(v.startswith("-Wl,-rpath,") for v in link)==1
link=["-Wl,-rpath,$ORIGIN" if v.startswith("-Wl,-rpath,") else v for v in link]
subprocess.run([*link,"-L"+str(profile),"-lea_probe"],cwd=old/"build",check=True)
r=re.findall(rpat,dyn(out))
if r!=["$ORIGIN"]:
    assert r==["$ORIGIN:"+str(root/"flightrl-puffer5.pmbvox/toolchain/lib")]
    subprocess.run(["cmake",f"-DLIBRARY={out}",f"-DOLD_RPATH={r[0]}",
                    "-P",str(old/"normalize_rpath.cmake")],check=True)
assert re.findall(rpat,dyn(out))==["$ORIGIN"]
assert libs(out)==libs(actual/"libni26_runtime.so")|{ea.name}
out.replace(profile/"libni26_runtime.so")
print("NATIVE_EA_CORE_LINK_PASS",flush=True)
