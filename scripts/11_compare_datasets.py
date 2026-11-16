"""Step 6b: one figure comparing NFCorpus and SCIDOCS side by side (reads results/<dataset>/).

  python scripts/11_compare_datasets.py

Left:  attack AUC per defense on each dataset (lower = better defense; 0.5 = coin flip).
Right: attack sessions flagged vs real honest (focused) sessions flagged, per defense.
Same colors as the other figures; circle = NFCorpus, square = SCIDOCS.
Output: results/comparison/fig6_datasets.png (+ .svg) and comparison_table.csv
"""
import json

import matplotlib.pyplot as plt
import pandas as pd

from ragmia import config as C

DATASETS = {"nfcorpus": ("NFCorpus", "o"), "scidocs": ("SCIDOCS", "s")}
DEFENSES = ["none", "mirabel", "session", "mirabel_session"]
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"none": MUTED, "mirabel": "#2a78d6", "session": "#eb6834", "mirabel_session": "#1baf7a"}
NAME = {"none": "No defense", "mirabel": "Mirabel", "session": "Session (ours)",
        "mirabel_session": "Mirabel + memory (ours)"}
OUT = C.ROOT / "results" / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": BASELINE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
})

rows = []
for ds in DATASETS:
    res = C.ROOT / "results" / ds
    honest = pd.read_csv(res / "honest_sessions.csv").set_index("defense")
    sweep = pd.read_csv(res / "session_sweep.csv")
    k2 = sweep[(sweep["window"] == 5) & (sweep["k"] == 2)].iloc[0]
    for d in DEFENSES:
        m = json.loads((res / f"attack_{d}_metrics.json").read_text())
        rows.append({"dataset": ds, "defense": d, "auc": m["auc"], "auc_ci_low": m["auc_95ci"][0],
                     "auc_ci_high": m["auc_95ci"][1], "tpr@1%fpr": m["tpr@1%fpr"],
                     "attack_sessions_flagged": honest.loc[d, "attack_sessions_flagged"],
                     "honest_focused_flagged": honest.loc[d, "focused_sessions_flagged"],
                     "honest_browsing_flagged": honest.loc[d, "browsing_sessions_flagged"]})
    rows.append({"dataset": ds, "defense": "session N=5 K=2",
                 "attack_sessions_flagged": k2["attack_sessions_flagged"],
                 "honest_focused_flagged": k2["focused_sessions_flagged"],
                 "honest_browsing_flagged": k2["browsing_sessions_flagged"]})
df = pd.DataFrame(rows)
df.to_csv(OUT / "comparison_table.csv", index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1, 1.15]})

# ---- Left: AUC dumbbells, one row per defense ----
for i, d in enumerate(DEFENSES):
    y = len(DEFENSES) - 1 - i
    pts = df[(df["defense"] == d)].set_index("dataset")
    ax1.plot([pts.loc["nfcorpus", "auc"], pts.loc["scidocs", "auc"]], [y, y], color=COLOR[d], lw=2, alpha=0.5)
    for ds, (label, marker) in DATASETS.items():
        ax1.errorbar(pts.loc[ds, "auc"], y, xerr=[[pts.loc[ds, "auc"] - pts.loc[ds, "auc_ci_low"]],
                     [pts.loc[ds, "auc_ci_high"] - pts.loc[ds, "auc"]]], fmt="none", ecolor=BASELINE, elinewidth=1)
        ax1.scatter(pts.loc[ds, "auc"], y, s=70, marker=marker, color=COLOR[d], edgecolor=SURFACE,
                    linewidth=1.5, zorder=3)
ax1.axvline(0.5, color=INK_2, lw=1)
ax1.text(0.505, len(DEFENSES) - 0.55, "coin flip", fontsize=8, color=INK_2)
ax1.set_yticks(range(len(DEFENSES)), [NAME[d] for d in reversed(DEFENSES)])
ax1.set_xlim(0.45, 1.02)
ax1.set_ylim(-0.6, len(DEFENSES) - 0.4)
ax1.set_xlabel("Attack AUC with 95% CI (lower = better defense)")
ax1.set_title("Same ranking on both datasets")
ax1.grid(axis="y", visible=False)

# ---- Right: detection vs honest cost ----
for d in ["mirabel", "session", "mirabel_session", "session N=5 K=2"]:
    key = "session" if d.startswith("session") else d
    for ds, (label, marker) in DATASETS.items():
        r = df[(df["dataset"] == ds) & (df["defense"] == d)].iloc[0]
        size = 260 if d == "mirabel" else 70   # Mirabel drawn larger so a point it shares stays visible
        ax2.scatter(r["honest_focused_flagged"] * 100, r["attack_sessions_flagged"] * 100, s=size, marker=marker,
                    color=COLOR[key], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                    alpha=0.55 if d == "session N=5 K=2" else 1)
notes = [("Session, K=3 (ours)", 6, 74, "left"), ("Session, K=2 (ours)", 13, 96, "left"),
         ("Mirabel + memory (ours)", 91, 96, "right"), ("Mirabel", 91, 83, "right")]
for text, x, y, ha in notes:
    ax2.text(x, y, text, fontsize=8.5, color=INK, ha=ha, va="center")
ax2.text(91, 91, "on SCIDOCS both sit at the\nsame point (larger blue ring)", fontsize=7.5, color=INK_2,
         ha="right", va="center")
ax2.text(2, 44, "ideal = top-left: catch every attacker, flag no honest user", fontsize=7.5, color=INK_2)
for ds, (label, marker) in DATASETS.items():
    ax2.scatter([], [], marker=marker, color=MUTED, s=55, label=label)
ax2.legend(frameon=False, loc="lower right", fontsize=8.5)
ax2.set_xlim(-4, 104)
ax2.set_ylim(40, 104)
ax2.set_xlabel("Real honest sessions flagged (%), focused users")
ax2.set_ylabel("Attack sessions flagged (%)")
ax2.set_title("Session rule: catches attacks, leaves honest users alone")

fig.tight_layout(w_pad=3)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"fig6_datasets.{ext}", dpi=200, bbox_inches="tight")
print("saved", OUT / "fig6_datasets.png")
print("saved", OUT / "comparison_table.csv")
print()
print(df.round(3).to_string(index=False))