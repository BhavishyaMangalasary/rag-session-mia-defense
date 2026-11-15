"""Step 6: figures for the report and slides (reads results/, writes results/figures/).

  python scripts/10_make_figures.py

Every condition keeps the same color in every figure:
  no defense = gray, Mirabel = blue, session (ours) = orange, Mirabel + memory (ours) = aqua.
Each figure is saved as PNG (for slides/docs) and SVG (sharp in print).
"""
import json

import matplotlib.pyplot as plt
import pandas as pd

from ragmia import config as C

OUT = C.RESULTS_DIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (light mode) + neutral tokens
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"none": MUTED, "mirabel": "#2a78d6", "session": "#eb6834", "mirabel_session": "#1baf7a"}
NAME = {"none": "No defense", "mirabel": "Mirabel", "session": "Session (ours)",
        "mirabel_session": "Mirabel + memory (ours)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": BASELINE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / f"{name}.png")


def spread(ys, gap):
    """Nudge end-of-line label positions apart so they never overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = list(ys)
    for a, b in zip(order, order[1:]):
        if pos[b] - pos[a] < gap:
            pos[b] = pos[a] + gap
    return pos


def end_labels(ax, xs_end, series, gap):
    """Direct labels at the right end of each line, in ink (a colored dot carries identity)."""
    ys = [s[1] for s in series]
    for (key, y, text), y_lab in zip(series, spread(ys, gap)):
        ax.annotate(text, xy=(xs_end, y), xytext=(xs_end + 0.12, y_lab), va="center", fontsize=9,
                    color=INK, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.8) if abs(y_lab - y) > 1e-9 else None)


metrics = {k: json.loads((C.RESULTS_DIR / f"attack_{k}_metrics.json").read_text())
           for k in ["none", "mirabel", "session", "mirabel_session"]}

# ---------- Figure 1: attack AUC per condition, with 95% CI ----------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
keys = list(metrics)
aucs = [metrics[k]["auc"] for k in keys]
lo = [metrics[k]["auc"] - metrics[k]["auc_95ci"][0] for k in keys]
hi = [metrics[k]["auc_95ci"][1] - metrics[k]["auc"] for k in keys]
ax.bar(range(len(keys)), aucs, width=0.5, color=[COLOR[k] for k in keys], edgecolor=SURFACE, linewidth=2)
ax.errorbar(range(len(keys)), aucs, yerr=[lo, hi], fmt="none", ecolor=INK_2, elinewidth=1, capsize=4)
for i, k in enumerate(keys):
    ax.text(i, metrics[k]["auc_95ci"][1] + 0.02, f"{aucs[i]:.3f}", ha="center", fontsize=10, color=INK)
ax.axhline(0.5, color=INK_2, lw=1)
ax.set_xlim(-0.5, len(keys) + 0.15)
ax.text(len(keys) + 0.13, 0.512, "coin flip\n(0.5)", ha="right", va="bottom", fontsize=8, color=INK_2)
ax.set_xticks(range(len(keys)), ["No defense", "Mirabel", "Session\n(ours)", "Mirabel + memory\n(ours)"])
ax.set_ylim(0.4, 1.08)
ax.set_ylabel("Attack AUC (lower = better defense)")
ax.set_title("How well MEntA still identifies members, per defense")
ax.grid(axis="x", visible=False)
save(fig, "fig1_attack_auc")

# ---------- Figure 2: who gets flagged ----------
honest = pd.read_csv(C.RESULTS_DIR / "honest_sessions.csv").set_index("defense")
sim = pd.read_csv(C.RESULTS_DIR / "simulated_sessions.csv").set_index("defense")
groups = ["Attack sessions\n(MEntA)", "Honest: real users,\nfocused", "Honest: real users,\nbrowsing",
          "Honest: deep-dive\non one paper"]
defs = ["mirabel", "session", "mirabel_session"]
vals = {d: [honest.loc[d, "attack_sessions_flagged"], honest.loc[d, "focused_sessions_flagged"],
            honest.loc[d, "browsing_sessions_flagged"], sim.loc[d, "honest deep-dive"]] for d in defs}
fig, ax = plt.subplots(figsize=(7.2, 3.8))
w = 0.24
for j, d in enumerate(defs):
    xs = [i + (j - 1) * (w + 0.02) for i in range(len(groups))]
    ax.bar(xs, [v * 100 for v in vals[d]], width=w, color=COLOR[d], edgecolor=SURFACE, linewidth=2,
           label=NAME[d])
    for x, v in zip(xs, vals[d]):
        ax.text(x, v * 100 + 2, f"{v * 100:.0f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(len(groups)), groups)
ax.set_ylim(0, 112)
ax.set_ylabel("Sessions flagged (%)")
ax.set_title("Who gets flagged: attackers vs honest users")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax.grid(axis="x", visible=False)
save(fig, "fig2_who_gets_flagged")

# ---------- Figure 3: query budget ----------
qb = pd.read_csv(C.RESULTS_DIR / "query_budget.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.6))
series = []
for k in ["none", "session", "mirabel"]:
    s = qb[qb["condition"] == k]
    ax.plot(s["queries"], s["auc"], color=COLOR[k], marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    series.append((k, float(s["auc"].iloc[-1]), NAME[k]))
end_labels(ax, 5, series, gap=0.035)
ax.axvline(2.5, color=BASELINE, lw=1)
ax.text(2.45, 0.6, "session defense (K=3)\nstarts hiding here", ha="right", va="bottom", fontsize=8, color=INK_2)
ax.set_xticks(range(1, 6))
ax.set_xlim(0.8, 5.2)
ax.set_ylim(0.55, 1.02)
ax.set_xlabel("Answers the attacker has seen")
ax.set_ylabel("Attack AUC")
ax.set_title("One answer is almost enough: AUC 0.92 after a single query")
save(fig, "fig3_query_budget")

# ---------- Figure 4: adaptive attacker ----------
ad = pd.read_csv(C.RESULTS_DIR / "adaptive_attacker.csv")
ad = ad[ad["prefix"]]
plot = [("none", "none"), ("mirabel", "mirabel"), ("session", "session"),
        ("mirabel_session", "mirabel_session")]
fig, ax = plt.subplots(figsize=(6.8, 3.8))
series = []
for name, key in plot:
    s = ad[ad["defense"] == name].sort_values("decoys_per_question")
    ax.plot(s["decoys_per_question"], s["target_leaked"], color=COLOR[key], marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    series.append((key, float(s["target_leaked"].iloc[-1]), NAME[key]))
end_labels(ax, 3, series, gap=0.32)
ax.set_xticks(range(4), ["0\n(5 queries)", "1\n(10)", "2\n(15)", "3\n(20)"])
ax.set_xlim(-0.2, 3.2)
ax.set_ylim(0, 5.2)
ax.set_xlabel("Decoy questions inserted after each attack question (total queries)")
ax.set_ylabel("Target leaked into context\n(of 5 attack questions)")
ax.set_title("Adaptive attacker: decoys evade the session rule, not Mirabel + memory")
save(fig, "fig4_adaptive_attacker")

# ---------- Figure 5: trade-off (attack detection vs honest false alarms) ----------
sweep = pd.read_csv(C.RESULTS_DIR / "session_sweep.csv")
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.scatter(sweep["focused_sessions_flagged"] * 100, sweep["attack_sessions_flagged"] * 100, s=55,
           color=COLOR["session"], edgecolor=SURFACE, linewidth=1.5, zorder=3, label=NAME["session"] + ", all (N, K)")
for _, r in sweep.iterrows():
    offsets = {(5, 3): (8, -3), (5, 5): (8, -3), (3, 2): (10, -14)}
    if (r["window"], r["k"]) in offsets:
        ax.annotate(f"N={r['window']:.0f}, K={r['k']:.0f}",
                    (r["focused_sessions_flagged"] * 100, r["attack_sessions_flagged"] * 100),
                    xytext=offsets[(r["window"], r["k"])], textcoords="offset points", fontsize=8, color=INK)
for d in ["mirabel", "mirabel_session"]:
    x, y = honest.loc[d, "focused_sessions_flagged"] * 100, honest.loc[d, "attack_sessions_flagged"] * 100
    ax.scatter([x], [y], s=70, color=COLOR[d], edgecolor=SURFACE, linewidth=1.5, zorder=3, label=NAME[d])
    ax.annotate(NAME[d], (x, y), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                fontsize=8, color=INK)
ax.annotate("ideal: catch every attacker,\nbother no honest user", (0, 100), xytext=(18, 52),
            textcoords="data", fontsize=8, color=INK_2,
            arrowprops=dict(arrowstyle="->", color=BASELINE, lw=1))
ax.set_xlim(-4, 104)
ax.set_ylim(40, 104)
ax.set_xlabel("Honest focused sessions flagged (%)  - lower is better")
ax.set_ylabel("Attack sessions flagged (%)  - higher is better")
ax.set_title("Detection trade-off on real NFCorpus users")
ax.legend(frameon=False, loc="lower right", fontsize=8)
save(fig, "fig5_tradeoff")

# ---------- Table view (same numbers, for the report appendix / accessibility) ----------
table = pd.DataFrame([{
    "condition": NAME[k], "auc": metrics[k]["auc"], "auc_ci_low": metrics[k]["auc_95ci"][0],
    "auc_ci_high": metrics[k]["auc_95ci"][1], "tpr@1%fpr": metrics[k]["tpr@1%fpr"],
    "tpr@10%fpr": metrics[k]["tpr@10%fpr"],
    "attack_sessions_flagged": honest.loc[k, "attack_sessions_flagged"],
    "honest_focused_flagged": honest.loc[k, "focused_sessions_flagged"],
    "honest_browsing_flagged": honest.loc[k, "browsing_sessions_flagged"],
} for k in keys])
table.to_csv(OUT / "summary_table.csv", index=False)
print("saved", OUT / "summary_table.csv")