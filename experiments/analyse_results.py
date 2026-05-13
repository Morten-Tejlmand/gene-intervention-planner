"""
Result Analysis
===============
Loads summary CSVs from all experiments and prints a unified report.

Run from project root:
    python experiments/analyse_results.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ARTIFACTS = Path("artifacts")
SEP = "-" * 70


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def main() -> None:
    print("=" * 70)
    print("  Gene Intervention Planner — Experiment Results")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Exp 01 — Baseline model comparison
    # ------------------------------------------------------------------
    section("Exp 01  Baseline Neural-Relevance Prediction (5-fold CV)")
    e1 = pd.read_csv(ARTIFACTS / "exp01_baseline/cv_results.csv")
    cols = ["Model", "roc_auc", "roc_auc_std", "f1", "f1_std", "average_precision", "average_precision_std"]
    print(fmt(e1[[c for c in cols if c in e1.columns]]))
    best = e1.loc[e1["average_precision"].idxmax()]
    print(f"\n  Best: {best['Model']}  AP={best['average_precision']:.3f}  AUROC={best['roc_auc']:.3f}")

    # ------------------------------------------------------------------
    # Exp 02 — Feature set comparison
    # ------------------------------------------------------------------
    section("Exp 02  Feature Set Comparison (RF, 5-fold CV)")
    e2 = pd.read_csv(ARTIFACTS / "exp02_feature_comparison/comparison_results.csv")
    cols2 = ["FeatureSet", "n_features", "roc_auc", "roc_auc_std", "average_precision", "average_precision_std"]
    print(fmt(e2[[c for c in cols2 if c in e2.columns]]))
    best2 = e2.loc[e2["average_precision"].idxmax()]
    gain = best2["average_precision"] / e2["average_precision"].iloc[0]
    print(f"\n  Best: {best2['FeatureSet']}  AP={best2['average_precision']:.3f}  ({gain:.1f}x over biophysics-only)")

    # ------------------------------------------------------------------
    # Exp 03 — AL strategies vs random
    # ------------------------------------------------------------------
    section("Exp 03  Active Learning vs Random (LR, 20 rounds × 15 queries, 10 trials)")
    e3 = pd.read_csv(ARTIFACTS / "exp03_al_vs_random/summary.csv")
    cols3 = ["strategy", "final_ap_mean", "final_ap_std", "final_recall_mean", "final_recall_std"]
    print(fmt(e3[[c for c in cols3 if c in e3.columns]]))
    best3 = e3.loc[e3["final_recall_mean"].idxmax()]
    rand3 = e3[e3["strategy"] == "random"].iloc[0]
    recall_lift = best3["final_recall_mean"] - rand3["final_recall_mean"]
    print(f"\n  Best recall: {best3['strategy']}  recall={best3['final_recall_mean']:.3f}  "
          f"(+{recall_lift:.3f} vs random)")

    # ------------------------------------------------------------------
    # Exp 04 — Hybrid acquisition comparison
    # ------------------------------------------------------------------
    section("Exp 04  Acquisition Strategy Comparison (LR, 20 rounds × 15 queries, 10 trials)")
    e4 = pd.read_csv(ARTIFACTS / "exp04_hybrid_acquisition/summary.csv")
    cols4 = ["strategy", "final_ap", "final_ap_std", "final_recall", "auc_ap_curve"]
    print(fmt(e4[[c for c in cols4 if c in e4.columns]]))
    best4 = e4.loc[e4["final_recall"].idxmax()]
    print(f"\n  Best recall: {best4['strategy']}  recall={best4['final_recall']:.3f}  "
          f"AP={best4['final_ap']:.3f}")

    # ------------------------------------------------------------------
    # Exp 05 — Pair prioritization vs random
    # ------------------------------------------------------------------
    section("Exp 05  Gene Pair Prioritization — Top-K Anchor Hit Rate")
    e5 = pd.read_csv(ARTIFACTS / "exp05_interaction_prioritization/topk_evaluation.csv")
    print(fmt(e5))
    best_k = e5.loc[e5["enrichment_fold"].idxmax()]
    print(f"\n  Best enrichment: K={int(best_k['K'])}  "
          f"model={best_k['model_rate']:.3f}  random={best_k['random_rate']:.3f}  "
          f"fold={best_k['enrichment_fold']:.2f}x")

    # ------------------------------------------------------------------
    # Exp 06 — PU Bagging vs standard RF
    # ------------------------------------------------------------------
    section("Exp 06  PU Bagging vs Standard RF (10 trials)")
    e6 = pd.read_csv(ARTIFACTS / "exp06_pu_learning/summary.csv")
    cols6 = ["model", "final_ap_mean", "final_ap_std", "final_recall_mean", "final_recall_std"]
    print(fmt(e6[[c for c in cols6 if c in e6.columns]]))
    pu = e6[e6["model"] == "pu_bagging"].iloc[0]
    rf = e6[e6["model"] == "standard_rf"].iloc[0]
    print(f"\n  PU Bagging vs RF: AP +{pu['final_ap_mean'] - rf['final_ap_mean']:.3f}  "
          f"recall +{pu['final_recall_mean'] - rf['final_recall_mean']:.3f}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    section("Summary — Key Metric per Experiment")
    summary_rows = [
        ("Exp01 LR baseline",    "Avg Precision (CV)",     f"{e1.loc[e1['average_precision'].idxmax(), 'average_precision']:.3f}"),
        ("Exp02 D_+functional",  "Avg Precision (CV)",     f"{e2.loc[e2['average_precision'].idxmax(), 'average_precision']:.3f}"),
        ("Exp03 uncertainty",    "Final recall@budget",    f"{e3.loc[e3['final_recall_mean'].idxmax(), 'final_recall_mean']:.3f}"),
        ("Exp04 neural_score",   "Final recall@budget",    f"{e4.loc[e4['final_recall'].idxmax(), 'final_recall']:.3f}"),
        ("Exp05 pair rank",      "Enrichment fold @ best K", f"{e5['enrichment_fold'].max():.2f}x"),
        ("Exp06 PU Bagging",     "AP vs standard RF",      f"{pu['final_ap_mean']:.3f} vs {rf['final_ap_mean']:.3f}"),
    ]
    header = f"{'Experiment':<28} {'Metric':<28} {'Value'}"
    print(header)
    print("-" * len(header))
    for exp, metric, val in summary_rows:
        print(f"{exp:<28} {metric:<28} {val}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
