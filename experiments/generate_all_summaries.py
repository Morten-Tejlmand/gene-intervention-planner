"""
Combined summary figure for all experiments.
Run from project root: python experiments/generate_all_summaries.py
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

try:
    from pub_style import apply_pub_style
    apply_pub_style()
except ImportError:
    pass

ART = Path("artifacts")
OUT = ART / "all_experiments_summary.png"
ART.mkdir(exist_ok=True)

CR = "#888888"
CB = "#2196F3"
CO = "#FF9800"
CG = "#4CAF50"
CP = "#E91E63"
CV = "#9C27B0"
FEAT_COLORS = [CB, CO, CG, CP]


def shade(ax, x, df, color, label):
    m = df["ap_mean"].values
    s = df["ap_std"].values
    ax.plot(x, m, color=color, label=label, linewidth=1.6)
    ax.fill_between(x, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), alpha=0.18, color=color)


def shade_recall(ax, x, df, color, label):
    m = df["recall_mean"].values
    s = df["recall_std"].values
    ax.plot(x, m, color=color, label=label, linewidth=1.6)
    ax.fill_between(x, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), alpha=0.18, color=color)


def note(ax, text):
    ax.axis("off")
    ax.text(0.05, 0.96, text, transform=ax.transAxes, fontsize=8.5, va="top",
            linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f9f9f9",
                      edgecolor="#cccccc", alpha=0.95))


def budget_axis(df, n_seed, query_size):
    return n_seed + np.arange(len(df)) * query_size


def safe_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
exp01   = safe_csv(ART / "exp01_baseline/cv_results.csv")
exp02   = safe_csv(ART / "exp02_feature_comparison/comparison_results.csv")

e03_sum  = safe_csv(ART / "exp03_al_vs_random/summary.csv")
e03_rand = safe_csv(ART / "exp03_al_vs_random/curves_random.csv")
e03_unc  = safe_csv(ART / "exp03_al_vs_random/curves_uncertainty.csv")
e03_qbc  = safe_csv(ART / "exp03_al_vs_random/curves_qbc.csv")

e04_sum  = safe_csv(ART / "exp04_hybrid_acquisition/summary.csv")
e04_unc  = safe_csv(ART / "exp04_hybrid_acquisition/curves_uncertainty.csv")
e04_ns   = safe_csv(ART / "exp04_hybrid_acquisition/curves_neural_score.csv")
e04_hyb  = safe_csv(ART / "exp04_hybrid_acquisition/curves_hybrid.csv")

e05_topk = safe_csv(ART / "exp05_interaction_prioritization/topk_evaluation.csv")

ART_PHENO = ART / "active_learning_phenotype"
pheno    = safe_csv(ART_PHENO / "al_pheno_01_baseline/01_baseline_summary.csv")

beh_sum  = safe_csv(ART / "behavioural_al/summary.csv")
beh_rand = safe_csv(ART / "behavioural_al/curve_random.csv")
beh_mar  = safe_csv(ART / "behavioural_al/curve_margin.csv")
beh_ucb  = safe_csv(ART / "behavioural_al/curve_ucb.csv")

# Phenotype AL curves
PHENO_LIST = ["locomotion_variant", "egg_laying_defective", "paralyzed", "dauer_constitutive"]
ph_curves: dict[str, dict] = {}
for ph in PHENO_LIST:
    ph_curves[ph] = {
        s: safe_csv(ART_PHENO / f"al_pheno_02_al_vs_random/02_curves_{ph}_{s}.csv")
        for s in ["random", "uncertainty", "neural_score"]
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 46))
gs  = gridspec.GridSpec(11, 3, figure=fig, hspace=0.60, wspace=0.35)


# ── Row 0 : Exp01 Baseline ────────────────────────────────────────────────
ax = [fig.add_subplot(gs[0, c]) for c in range(3)]

if exp01 is not None:
    short = [m.replace("GradientBoosting","GBT").replace("LogisticRegression","LR")
               .replace("RandomForest","RF") for m in exp01["Model"]]
    mc = [CB, CO, CG]
    for a_idx, col, err, title, ylim in [
        (0, "roc_auc",           "roc_auc_std",           "AUROC", (0.70, 1.02)),
        (1, "average_precision", "average_precision_std", "AP",    (0.00, 0.28)),
    ]:
        a = ax[a_idx]
        bars = a.bar(short, exp01[col], yerr=exp01[err], color=mc, capsize=4, alpha=0.85)
        a.set(title=f"Exp01 — Baseline {title}", ylabel=title, ylim=ylim)
        for b, v, e in zip(bars, exp01[col], exp01[err]):
            a.text(b.get_x() + b.get_width()/2, v + e + ylim[1]*0.02,
                   f"{v:.3f}", ha="center", fontsize=8)

note(ax[2], (
    "Exp01: Baseline Models\n\n"
    "Target: is_neural  (~3% positive rate)\n"
    "Best model: RF   AUROC 0.918, AP 0.197\n\n"
    "AUROC misleading at 3% prevalence.\n"
    "Low AP reflects severe class imbalance.\n"
    "Establishes the need for active learning."
))


# ── Row 1 : Exp02 Feature Sets ────────────────────────────────────────────
ax = [fig.add_subplot(gs[1, c]) for c in range(3)]

if exp02 is not None:
    short_fs = ["A\nBiophys", "B\n+AA comp", "C\n+Evol", "D\n+GO/ESM2"]
    for a_idx, col, err, title, ylim in [
        (0, "roc_auc",           "roc_auc_std",           "AUROC", (0.50, 1.05)),
        (1, "average_precision", "average_precision_std", "AP",    (0.00, 0.23)),
    ]:
        a = ax[a_idx]
        bars = a.bar(short_fs, exp02[col], yerr=exp02[err],
                     color=FEAT_COLORS, capsize=4, alpha=0.85)
        a.set(title=f"Exp02 — Feature Set {title}", ylabel=title, ylim=ylim)
        for b, v, e in zip(bars, exp02[col], exp02[err]):
            a.text(b.get_x() + b.get_width()/2, v + e + ylim[1]*0.02,
                   f"{v:.3f}", ha="center", fontsize=8)

note(ax[2], (
    "Exp02: Feature Set Comparison\n\n"
    "  A biophysics:    AUROC 0.717  AP 0.017\n"
    "  B +AA comp:      AUROC 0.738  AP 0.024\n"
    "  C +evolutionary: AUROC 0.755  AP 0.048\n"
    "  D +GO/ESM2:      AUROC 0.924  AP 0.175\n\n"
    "GO/ESM2 embeddings give the largest\n"
    "single gain. Sequence-only features\n"
    "provide weak signal."
))


# ── Row 2 : Exp03 AL vs Random ────────────────────────────────────────────
ax = [fig.add_subplot(gs[2, c]) for c in range(3)]

if all(d is not None for d in [e03_rand, e03_unc, e03_qbc]):
    for df, color, label in [(e03_rand, CR, "random"),
                              (e03_unc,  CB, "uncertainty"),
                              (e03_qbc,  CO, "QBC")]:
        b = budget_axis(df, 10, 15)
        shade(ax[0], b, df, color, label)
        shade_recall(ax[1], b, df, color, label)
    ax[0].set(title="Exp03 — Average Precision vs Budget", xlabel="Labeled genes", ylabel="AP")
    ax[1].set(title="Exp03 — Recall vs Budget", xlabel="Labeled genes", ylabel="Recall")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)

    if e03_sum is not None:
        idx = e03_sum.set_index("strategy")
        aulc = [idx.loc[s, "aulc"] for s in ["random","uncertainty","qbc"] if s in idx.index]
        labs = ["random","uncert.","QBC"][:len(aulc)]
        bars = ax[2].bar(labs, aulc, color=[CR, CB, CO][:len(aulc)], alpha=0.85)
        for b, v in zip(bars, aulc):
            ax[2].text(b.get_x() + b.get_width()/2, v + 0.04,
                       f"{v:.2f}", ha="center", fontsize=8)
        ax[2].set(title="Exp03 — AULC", ylabel="AULC")

note(ax[2], (
    "Exp03: AL vs Random\n\n"
    "Uncertainty reaches recall 0.20\n"
    "vs random 0.015 — 13× better.\n"
    "QBC: recall 0.15\n\n"
    "AULC: uncertainty 5.63,\n"
    "      random 4.53,  QBC 4.50\n\n"
    "Model-guided selection clearly\n"
    "outperforms random on recall."
)) if e03_sum is None else None


# ── Row 3 : Exp04 Hybrid ─────────────────────────────────────────────────
ax = [fig.add_subplot(gs[3, c]) for c in range(3)]

if all(d is not None for d in [e04_unc, e04_ns, e04_hyb]):
    for df, color, label in [(e04_unc, CB, "uncertainty"),
                              (e04_ns,  CO, "neural_score"),
                              (e04_hyb, CG, "hybrid")]:
        b = budget_axis(df, 10, 15)
        shade(ax[0], b, df, color, label)
        shade_recall(ax[1], b, df, color, label)
    ax[0].set(title="Exp04 — Hybrid Strategies AP", xlabel="Labeled genes", ylabel="AP")
    ax[1].set(title="Exp04 — Hybrid Strategies Recall", xlabel="Labeled genes", ylabel="Recall")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)

    if e04_sum is not None:
        idx = e04_sum.set_index("strategy")
        strats = ["uncertainty","neural_score","hybrid"]
        aulc = [idx.loc[s, "aulc"] for s in strats if s in idx.index]
        labs = [s.replace("neural_score","neural\nscore") for s in strats[:len(aulc)]]
        bars = ax[2].bar(labs, aulc, color=[CB, CO, CG][:len(aulc)], alpha=0.85)
        for b, v in zip(bars, aulc):
            ax[2].text(b.get_x() + b.get_width()/2, v + 0.02,
                       f"{v:.2f}", ha="center", fontsize=8)
        ax[2].set(title="Exp04 — AULC", ylabel="AULC")


# ── Row 4 : Exp05 Gene Interactions ───────────────────────────────────────
ax = [fig.add_subplot(gs[4, c]) for c in range(3)]

if e05_topk is not None:
    ax[0].plot(e05_topk["K"], e05_topk["model_rate"],  "o-",  color=CB, label="model",  lw=1.6)
    ax[0].plot(e05_topk["K"], e05_topk["random_rate"], "s--", color=CR, label="random", lw=1.4)
    ax[0].set(title="Exp05 — Top-K Hit Rate", xlabel="K", ylabel="Hit Rate")
    ax[0].legend(fontsize=8)

    col = [CB if v >= 1.0 else CO for v in e05_topk["enrichment_fold"]]
    ax[1].bar(e05_topk["K"].astype(str), e05_topk["enrichment_fold"], color=col, alpha=0.85)
    ax[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    for i, v in enumerate(e05_topk["enrichment_fold"]):
        ax[1].text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax[1].set(title="Exp05 — Enrichment Fold at K\n(≥ 1.0 = beats random)", ylabel="Fold")

note(ax[2], (
    "Exp05: Gene Interaction Ranking\n\n"
    "Pairwise scoring: biophysical\n"
    "complementarity (hydrophobic fit,\n"
    "charge, stability, conservation).\n\n"
    "Model < random at all K (enrich < 1).\n"
    "Biophysical proxies alone do not\n"
    "predict interaction hit rate.\n\n"
    "Network / expression features\n"
    "would likely improve this."
))


# ── Row 5 : Phenotype Baseline ────────────────────────────────────────────
ax = [fig.add_subplot(gs[5, c]) for c in range(3)]

if pheno is not None:
    PHENOS_ALL = ["locomotion_variant","egg_laying_defective",
                  "paralyzed","dauer_constitutive","mechanosensation_variant"]
    ph_best = pheno.groupby("phenotype")[["ap_mean","auroc_mean"]].max().reset_index()
    ph_best = ph_best.set_index("phenotype").reindex(PHENOS_ALL).reset_index()
    ph_labels = ["Locomotion","Egg-laying","Paralyzed","Dauer","Mechan."]
    ph_c = [CB, CO, CG, CP, CV]

    bars = ax[0].bar(ph_labels, ph_best["ap_mean"], color=ph_c, alpha=0.85)
    ax[0].set(title="Phenotype Baseline — Best AP", ylabel="AP")
    ax[0].tick_params(axis="x", labelsize=8, rotation=15)
    for b, v in zip(bars, ph_best["ap_mean"]):
        if not np.isnan(v):
            ax[0].text(b.get_x() + b.get_width()/2, v + 0.005,
                       f"{v:.3f}", ha="center", fontsize=8)

    bars = ax[1].bar(ph_labels, ph_best["auroc_mean"], color=ph_c, alpha=0.85)
    ax[1].set(title="Phenotype Baseline — Best AUROC", ylabel="AUROC", ylim=(0.5, 1.02))
    ax[1].tick_params(axis="x", labelsize=8, rotation=15)
    ax[1].axhline(0.5, color="black", linestyle="--", linewidth=0.7, alpha=0.4)
    for b, v in zip(bars, ph_best["auroc_mean"]):
        if not np.isnan(v):
            ax[1].text(b.get_x() + b.get_width()/2, v + 0.005,
                       f"{v:.3f}", ha="center", fontsize=8)

note(ax[2], (
    "Phenotype Baseline (5 phenotypes)\n\n"
    "  Locomotion:   AP 0.484  AUROC 0.932\n"
    "  Egg-laying:   AP 0.274  AUROC 0.941\n"
    "  Paralyzed:    AP 0.158  AUROC 0.932\n"
    "  Dauer:        AP 0.151  AUROC 0.868\n"
    "  Mechan.:      AP 0.065  AUROC 0.908\n\n"
    "Locomotion is the most learnable.\n"
    "AUROC high across all but misleading\n"
    "at low prevalence (< 1% pos)."
))


# ── Rows 6-9 : Phenotype AL learning curves (one row per phenotype) ────────
PH_TITLES = {
    "locomotion_variant":   "Locomotion variant (6.9%, n=1373)",
    "egg_laying_defective": "Egg-laying defective (1.1%, n=217)",
    "paralyzed":            "Paralyzed (0.8%, n=159)",
    "dauer_constitutive":   "Dauer constitutive (0.6%, n=123)",
}

for row, ph in enumerate(PHENO_LIST, start=6):
    ax = [fig.add_subplot(gs[row, c]) for c in range(3)]
    title = PH_TITLES[ph]

    for s, color in [("random", CR), ("uncertainty", CB), ("neural_score", CO)]:
        df = ph_curves[ph].get(s)
        if df is not None:
            b = budget_axis(df, 48, 15)
            shade(ax[0], b, df, color, s)
            m = df["enrich_mean"].values
            e = df["enrich_std"].values
            ax[1].plot(b, m, color=color, label=s, lw=1.6)
            ax[1].fill_between(b, np.clip(m - e, 0, None), m + e, alpha=0.18, color=color)

    ax[0].set(title=f"Pheno AL — AP: {title}", xlabel="Labeled genes", ylabel="AP")
    ax[1].set(title=f"Pheno AL — Enrichment: {title}", xlabel="Labeled genes",
              ylabel="Enrichment (×)")
    ax[1].axhline(1.0, color="black", linestyle="--", lw=0.8)
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)

    # Final AP bar (summary)
    final_ap   = {s: ph_curves[ph][s]["ap_mean"].iloc[-1]
                  for s in ["random","uncertainty","neural_score"]
                  if ph_curves[ph].get(s) is not None}
    if final_ap:
        bars = ax[2].bar(list(final_ap.keys()), list(final_ap.values()),
                         color=[CR, CB, CO][:len(final_ap)], alpha=0.85)
        ax[2].set(title=f"Final AP after 20 rounds\n{title}", ylabel="AP")
        ax[2].tick_params(axis="x", labelsize=8)
        for b, v in zip(bars, final_ap.values()):
            ax[2].text(b.get_x() + b.get_width()/2, v + 0.003,
                       f"{v:.3f}", ha="center", fontsize=8)


# ── Row 10 : Behavioural AL ───────────────────────────────────────────────
ax = [fig.add_subplot(gs[10, c]) for c in range(3)]

if all(d is not None for d in [beh_rand, beh_mar, beh_ucb]):
    for df, color, label in [(beh_rand, CR, "random"),
                              (beh_mar,  CO, "margin"),
                              (beh_ucb,  CB, "UCB")]:
        b = budget_axis(df, 20, 10)
        shade(ax[0], b, df, color, label)
        shade_recall(ax[1], b, df, color, label)
    ax[0].set(title="Behavioural AL — AP vs Budget\n(neural_behaviour_count ≥ 1, 12.4% pos)",
              xlabel="Labeled genes", ylabel="AP")
    ax[1].set(title="Behavioural AL — Recall vs Budget", xlabel="Labeled genes", ylabel="Recall")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)

    if beh_sum is not None:
        idx = beh_sum.set_index("strategy")
        strats_b = ["random","margin","ucb"]
        w = 0.35; x_b = np.arange(3)
        recs = [idx.loc[s,"final_rec_mean"] for s in strats_b if s in idx.index]
        aps  = [idx.loc[s,"final_ap_mean"]  for s in strats_b if s in idx.index]
        n    = len(recs)
        ax[2].bar(x_b[:n] - w/2, recs, w, label="Recall", color=[CR,CO,CB][:n], alpha=0.85)
        ax[2].bar(x_b[:n] + w/2, aps,  w, label="AP",     color=[CR,CO,CB][:n], alpha=0.45, hatch="//")
        ax[2].set_xticks(x_b[:n])
        ax[2].set_xticklabels(strats_b[:n])
        ax[2].set(title="Behavioural AL — Final Recall & AP", ylabel="Score")
        ax[2].legend(fontsize=8)
        for i, (r, a) in enumerate(zip(recs, aps)):
            ax[2].text(i - w/2, r + 0.001, f"{r:.4f}", ha="center", fontsize=7.5, rotation=90)
            ax[2].text(i + w/2, a + 0.001, f"{a:.3f}", ha="center", fontsize=7.5, rotation=90)

note(ax[2], (
    "Behavioural AL (new framing)\n\n"
    "Target: neural_behaviour_count >= 1\n"
    "12.4% positives (vs 3% before)\n\n"
    "Recall after 270 genes labeled:\n"
    "  random: 0.0135\n"
    "  margin: 0.0423  (+3×)\n"
    "  UCB:    0.0427  (+3×)\n\n"
    "AP similar across strategies;\n"
    "recall improvement is the signal.\n"
    "More rounds / budget needed."
)) if beh_sum is None else None


# ---------------------------------------------------------------------------
fig.suptitle(
    "C. elegans Gene Intervention Planner — All Experiments Summary",
    fontsize=15, fontweight="bold", y=1.003,
)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved -> {OUT}")
plt.close()
