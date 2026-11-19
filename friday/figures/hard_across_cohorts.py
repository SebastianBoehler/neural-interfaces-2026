"""Plot the evidence for hard waveform substitution and its direct RTX runtime."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from adaptation import interval
from native_adaptation import locked
from plot import BLUE, INK, MUTED, TEAL, save

HELD = "23859db5ca29af650a9e0bda6bf86971a214c02b98fb7f4a54374faeec6519d6"
HELD_A100 = "01cef60a3838299076d23d761e67cd2e00724c4996062ed6e1fb81a761805dae"
RETURNING = "811d15ace58f38726d91692c7729ab26c177054868d700677203fc6a28a511d4"
EXTERNAL = "9102f588e0f60772ab02d004e51bd686ad69c66ed0f8f641175ca926b9df25cf"
G = "c94bfff42f87d4ddd3832d00fb433f7243544eeca44bc7ebc753469a99ede861"
P10 = "ae050c0e7de043a43d5948c01201d96add7e8c0a1e6c89ba00f8f225c44582e3"


def plot(folder, direct_sha):
    held = locked(folder/"held.json",HELD)["orders"]
    a100 = locked(folder/"held-a100.json",HELD_A100)["orders"]
    returning = locked(folder/"returning.json",RETURNING)["pairings"]["20_80"]
    pool = locked(folder/"external.json",EXTERNAL)
    direct = locked(folder/"direct.json",direct_sha)["contrasts"]["hard_vs_mean_stock"]
    rows = (
        ("Held-new · within0", held["within0"]["contrasts"]["hard_minus_stock"]),
        ("Held-new · full0", held["full0"]["contrasts"]["hard_minus_stock"]),
        ("Returning · P04/P10 at 20/80", returning),
        ("External · shared inputs", dict(pp=pool["pooled_pp"],paired80=pool["paired80"])),
        ("External · exact ZIPs", direct),
    )
    fig, (accuracy,runtime)=plt.subplots(1,2,figsize=(12,4.8),layout="constrained",
                                        gridspec_kw={"width_ratios":[1.40,1]})
    fig.get_layout_engine().set(rect=(.020,.13,.97,.78))
    fig.suptitle("Hard waveform accuracy and direct RTX runtime",x=.02,ha="left",
                 fontsize=18,color=INK,weight="bold")
    accuracy.axvline(0,color=MUTED,lw=.9,ls="--")
    for row,(_,value) in enumerate(rows):
        y=row-.13 if row<2 else row
        interval(accuracy,value,y,BLUE,"H100 fits and exact package" if row==0 else None)
    for row,order in enumerate(("within0","full0")):
        count=a100[order]["contrasts"]["hard_minus_stock"]
        interval(accuracy,count,row+.13,TEAL,"A100 fit replication" if row==0 else None)
    accuracy.legend(loc="lower left",bbox_to_anchor=(0,1.01),ncol=2,frameon=False,fontsize=8)
    accuracy.axhline(1.5,color="#d7e1e9",lw=1)
    accuracy.axhline(2.5,color="#d7e1e9",lw=1)
    accuracy.set_yticks(range(len(rows)),[name for name,_ in rows])
    accuracy.set(ylim=(4.6,-.55),xlabel="Balanced accuracy difference (percentage points)")
    runtime.set_title("Warm, paired worker runs",loc="left",fontsize=11,color=MUTED)
    runtime.axvline(1,color=MUTED,lw=.9,ls="--")
    runtime.axvline(1.12,color=MUTED,lw=.8,ls=":",label="Mean limit")
    runtime.axvline(1.15,color="#a8b6c4",lw=.8,ls=":",label="Worst-pair limit")
    for row,(name,digest) in enumerate((("runtime-g.json",G),("runtime-p10.json",P10))):
        record=locked(folder/name,digest)["result"]
        runtime.plot(record["paired"],[row]*len(record["paired"]),"|",ms=20,color=MUTED)
        runtime.scatter(record["ratio"],row,s=44,color=TEAL,zorder=3,label="Mean ratio" if row==0 else None)
    runtime.set_yticks((0,1),("G · unrecognized","P10 · recognized"))
    runtime.set(xlim=(.99,1.164),ylim=(1.5,-.5),xlabel="Elapsed time relative to stock")
    runtime.legend(loc="lower left",frameon=False,fontsize=8)
    for ax in (accuracy,runtime):
        ax.spines[["top","right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x",color="#e1e7ee")
    fig.text(.023,.037,"Accuracy intervals resample people · external repeats use the same nine Scherer people · timings cover two fixtures",
             fontsize=8.9,color=MUTED)
    save(fig,folder,"hard-waveform-accuracy-runtime")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    p.add_argument("direct_sha")
    a=p.parse_args()
    plot(a.folder,a.direct_sha)
