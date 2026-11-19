"""Plot the same complete multiresolution contrasts on each tested instrument."""
import argparse
from pathlib import Path

from adaptation import interval, layout
from native_adaptation import locked
from plot import BLUE, GOLD, MUTED, TEAL, save


def plot(folder, out):
    held = locked(folder/"held.json",
                  "309c3f6d368b4b1034e963233d54df61cb372b62b1b07f289cd95e1df26e63c0")["orders"]
    external = locked(folder/"external.json",
                      "90880bdbc01f62e3264edd2b8c87e7f3ec3e4f873b53bb22bacbec5c16a29958")["contrasts"]
    groups = (
        ("Held-new · within0", held["within0"]["contrasts"],
         ("joint_minus_stock", "joint_minus_FBL2", "joint_minus_single_joint")),
        ("Held-new · full0", held["full0"]["contrasts"],
         ("joint_minus_stock", "joint_minus_FBL2", "joint_minus_single_joint")),
        ("External · Scherer", external,
         ("combo_minus_stock", "combo_minus_fbl2", "combo_minus_conventional_combo")),
    )
    fig, (ax,) = layout("The multiresolution combination was flat externally",width=11)
    labels = ("Versus cap250", "Versus FBLight", "Versus conventional + FBLight")
    for row, (_, contrasts, keys) in enumerate(groups):
        for key, color, offset, label in zip(keys, (BLUE,GOLD,TEAL),(-.18,0,.18),labels,strict=True):
            interval(ax,contrasts[key],row+offset,color,label if row==0 else None)
    ax.axhline(1.5,color="#d7e1e9",lw=1)
    ax.set_yticks(range(len(groups)),[name for name,_,_ in groups])
    ax.set(xlim=(-.49,1.07),ylim=(2.62,-.62))
    ax.legend(loc="lower left",bbox_to_anchor=(0,1.01),ncol=3,frameon=False,fontsize=9)
    fig.text(.028,.024,"Paired 80% intervals · 10 held-new people, H100 source fits · 9 reused Scherer people, A100 source fits",
             fontsize=8.5,color=MUTED)
    save(fig,out,"multiresolution-transfer")
    plot_selection(folder,out)


def plot_selection(folder,out):
    data = locked(folder/"audit.json",
                  "3c1645151cadd5a093e75420a7888e5aaffd0d2b16b9950ac41500a7a8ad7ec3")["all_vs_stock"]
    names = {
        "hard": "Hard waveform",
        "multi": "Multiresolution waveform",
        "single": "Conventional waveform",
        "fbl2": "Doubled FBLight",
        "fbl2_hard": "FBLight + hard",
        "fbl2_multi": "FBLight + multiresolution",
        "fbl2_single": "FBLight + conventional",
    }
    fig, (ax,) = layout("Conventional waveform led the external table",width=10.5)
    for y,(key,_) in enumerate(names.items()):
        result=data[key]
        lo,hi=result["simultaneous80"]
        ax.hlines(y,lo,hi,color="#a8b6c4",linewidth=3,label="Simultaneous 80%" if y==0 else None)
        interval(ax,result,y,BLUE,"Pairwise 80%" if y==0 else None)
    ax.set_yticks(range(len(names)),list(names.values()))
    ax.set(xlim=(-.75,.96),ylim=(6.53,-.53))
    ax.axhline(2.5,color="#d7e1e9",lw=1)
    ax.legend(loc="lower right",bbox_to_anchor=(1,1),ncol=2,frameon=False,fontsize=9)
    fig.text(.028,.024,"Retrospective full-system results · 9 reused Scherer people · 2 deliveries",
             fontsize=8.5,color=MUTED)
    save(fig,out,"external-waveform-selection")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    args=p.parse_args()
    plot(args.folder,args.folder)
