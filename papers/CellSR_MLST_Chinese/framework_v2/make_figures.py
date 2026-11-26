"""Rebuild the English figures in the revised CellSR paper outline."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "benchmark_outputs/cellsr1_revision"
sys.path.insert(0, str(HERE.parents[2] / "handoff"))
from horst_same_trial import subject_errors
INK, MUTED, TEAL, BLUE, OCHRE = "#192E3B", "#50626B", "#007468", "#3F61A0", "#9A591D"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#ABB9BD", "axes.axisbelow": True,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=230, facecolor="white", bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def mechanism():
    fig, ax = plt.subplots(figsize=(9.35, 5.8))
    fig.subplots_adjust(left=.02, right=.99, top=.99, bottom=.01)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")

    def label(x, y, s, size=10, color=INK, **kw):
        return ax.text(x, y, s, fontsize=size, color=color, va="center", **kw)

    def rule(x1, x2, y):
        ax.plot([x1, x2], [y, y], color="#C6D1D3", lw=.8)

    label(1, 96, "One coefficient in two different symbolic laws", 18, weight="bold")
    rule(1, 99, 90)
    label(1, 84, "a   Local expression search", 13, weight="bold")
    label(53, 84, "b   Declare a coefficient correspondence", 13, weight="bold")
    label(1, 77, "A symbolic cell carries a candidate expression.")
    label(1, 70, "Example features in two local laws", 9, MUTED)
    for y, r, nodes in [(61,"A",[(15,"z"),(25,"square"),(36,"sin")]), (52,"B",[(15,"z"),(25,"cos")])]:
        label(1, y, f"Law {r}", 11, weight="bold")
        boxes = []
        for x, op in nodes:
            text = label(x, y, op, 10, ha="center",
                         bbox=dict(boxstyle="round,pad=0.26",fc="#F0F3F5",ec="#B5C1C8",lw=.6))
            boxes.append(text.get_bbox_patch())
        for i in range(len(nodes) - 1):
            ax.annotate("", xy=(nodes[i + 1][0], y), xytext=(nodes[i][0], y),
                        arrowprops=dict(arrowstyle="->", lw=1, color=MUTED,
                                        patchA=boxes[i], patchB=boxes[i + 1], shrinkA=3, shrinkB=3))
    label(53, 77, "Each fitted law has three coefficient slots.")
    for y, r in [(61, "A"), (52, "B")]:
        label(53, y, f"Law {r}")
        for x, letter in [(67, "b"), (78, r"\theta"), (89, "a")]:
            shared = x == 78
            ax.add_patch(Rectangle((x-4.8, y-3.2), 9.6, 6.5, facecolor="#D8EFEB" if shared else "#F1F3F3", edgecolor="none"))
            label(x, y, f"${letter}_{r}$", 13, TEAL if shared else INK, ha="center")
    for x, s in [(67, "1"), (78, "x"), (89, r"$g_r(z)$")]:
        label(x, 69.5, s, 10, MUTED, ha="center")
    ax.plot([78, 78], [55.4, 57.8], color=TEAL, ls=":", lw=2)
    label(1, 44, r"Local form:  $y_r=b_r+\theta_r x_r+a_r g_r(z_r)$", 11)
    label(53, 44, r"Declared hypothesis: $\theta_A=\theta_B$.", 11, TEAL)
    rule(1, 99, 38)
    label(1, 32, "c   Fit under the proposed equality", 13, weight="bold")
    label(53, 32, "d   Choose on held-out observations", 13, weight="bold")
    label(1, 25, "One common column, four private columns.", 10)
    for x, s in zip([12, 19, 26, 33, 40], [r"$b_A$", r"$a_A$", r"$\theta$", r"$b_B$", r"$a_B$"]):
        label(x, 18.5, s, 10, TEAL if x == 26 else MUTED, ha="center")
    ax.add_patch(Rectangle((22.7, 1.5), 6.6, 13, facecolor="#D8EFEB", edgecolor="none"))
    for y, r, vals in [(11.5,"A",["1",r"$g_A$",r"$x_A$","0","0"]), (4.5,"B",["0","0",r"$x_B$","1",r"$g_B$"])]:
        label(1, y, f"Rows {r}", 9)
        for x, s in zip([12, 19, 26, 33, 40], vals):
            label(x, y, s, 11, TEAL if x == 26 else INK, ha="center")
    label(53, 24.5, "Compare the joint fit with independent fits.", 10)
    label(53, 17.7, "Require full joint rank, lower mean validation loss,", 10)
    label(53, 12.5, "and at most 5% higher loss for each law.", 10)
    label(53, 4.5, "Accept → one θ     |     Reject → separate θ estimates", 10, TEAL, weight="bold")
    save(fig, "fig1_mechanism")


def coefficient_gap():
    data = pd.read_csv(DATA / "coefficient_gap/summary.csv")
    gap = (data.theta - 1.2).round(1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.35, 3.55), layout="constrained")
    for col, label, color, marker in [("independent", "Independent", MUTED, "o"), ("joint", "Forced equality", BLUE, "s"),
                                      ("soft", "Soft sharing", OCHRE, "^"), ("CellSR", "CellSR", TEAL, "D")]:
        a.plot(gap, data[col], label=label, color=color, marker=marker, lw=1.6, ms=4)
    a.legend(frameon=False, loc="upper left")
    a.set_title("a   Target coefficient error", loc="left", weight="bold")
    a.set_ylabel("Mean absolute error")
    b.plot(gap, data.accepted / 128 * 100, color=TEAL, marker="D")
    for x, n in zip(gap, data.accepted):
        beside_slope = x in (0.1, 0.2, 0.4)
        b.annotate(f"{n}/128", (x, n/128*100), xytext=(4 if beside_slope else 0, 8),
                   textcoords="offset points", ha="left" if beside_slope else "center", fontsize=8)
    b.set_ylim(-3, 65)
    b.margins(x=.09)
    b.set_title("b   CellSR sharing decisions", loc="left", weight="bold")
    b.set_ylabel("Accepted equalities (%)")
    for ax in (a, b):
        ax.set_xlabel("Auxiliary minus target coefficient")
        ax.set_xticks(gap)
        ax.grid(axis="y", color="#E1E7E7")
    save(fig, "fig2_gap")


def walking():
    data = pd.read_csv(DATA / "external_case_comparison_v1/gait/scored.csv")
    given = data[(data.kind=="one") & data.method.isin(["independent","fixed_B","fixed_C"])].copy()
    given["assignment"] = np.where(given.method=="independent","Independent",np.where(given.graph==given.truth,"Same person","Other person"))
    subjects = given.groupby(["cluster","assignment"]).mae.mean().unstack()[["Independent","Other person","Same person"]]
    assert len(subjects)==56
    ids, errors = subject_errors()
    assert list(subjects.index) == ids
    np.testing.assert_allclose(subjects[["Independent","Same person"]], errors[:,[0,2]])
    subjects["Impulse only"] = errors[:,1]
    ext = pd.read_csv(DATA / "matched_false_share/comparison.csv").set_index("method").loc[["B","NN_frozen","NN_equal"]]
    fig, (a,b) = plt.subplots(1,2,figsize=(9.35,3.8),layout="constrained",gridspec_kw={"width_ratios":[1,1.13]})
    a.plot(range(4), subjects.T, color="#A7B8BE",alpha=.43, lw=.65)
    mean = subjects.mean()
    a.plot(range(4),mean,color=TEAL,marker="o",lw=2.7)
    for i,v in enumerate(mean):
        label_height = 11 if i >= 2 else 8
        a.annotate(f"{v:.2f}",(i,v),xytext=(0,label_height),textcoords="offset points",ha="center",color=TEAL,weight="bold",bbox=dict(fc="white",alpha=.82,ec="none",pad=.3))
    a.set_xticks(range(4),["Work–energy\nonly","Joint\nother person","Joint\nsame person","Impulse only\nsame person"])
    a.set_xlim(-.35,3.35);a.set_ylim(bottom=0)
    a.set_title("a   Body mass estimates · Horst",loc="left",fontsize=12,weight="bold")
    a.set_ylabel("Target mass MAE (kg)");a.set_xlabel("Thin lines: 56 people   ·   Bold line: mean")
    a.grid(axis="y",color="#E1E7E7")
    b.set_title("b   Automatic choice · Antwerp",loc="left",fontsize=12,weight="bold")
    b.set_xlabel("Any false link with no valid donor (%)")
    b.set_ylabel("Exact link and lower error\nwith one valid donor (%)")
    b.set(xlim=(33,49),ylim=(8,33))
    for (method,row),name,color,marker,dy in zip(ext.iterrows(),["CellSR","Nearest, frozen","Nearest, risk-matched"],[TEAL,BLUE,BLUE],["o","^","s"],[-1,1,-1]):
        x,y = row.wrong_rate*100,row.both*100
        b.scatter(x,y,s=55,c=color,marker=marker,zorder=3)
        b.annotate(name,(x,y),xytext=(6,dy*7),textcoords="offset points",fontsize=8,color=color)
    x=ext.loc["B","wrong_rate"]*100
    b.plot([x,x],ext.loc[["B","NN_equal"],"both"]*100,ls=":",color=MUTED)
    b.grid(color="#E1E7E7")
    save(fig,"fig3_walking")


if __name__ == "__main__":
    mechanism()
    coefficient_gap()
    walking()
