"""
Result Analysis
===============
Loads summary CSVs from the 4 behavioural-AL experiments and prints a
unified report.

Run from project root:
    python experiments/analyse_results.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path("artifacts/behavioural_al")
SEP = "-" * 70


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def main() -> None:
    print("=" * 70)
    print("  Gene Intervention Planner — Experiment Results")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Exp 01 — Baseline model comparison
    # ------------------------------------------------------------------
    section("Exp 01  Baseline Model Comparison (5-fold CV, neural_behaviour_count >= 1)")
    e1 = pd.read_csv(BASE / "exp01_baseline/cv_results.csv")
    print(e1[["model", "auroc", "auroc_std", "ap", "ap_std"]].to_string(index=False))
    best1 = e1.loc[e1["ap"].idxmax()]
    print(f"\n  Best: {best1['model']}  AP={best1['ap']:.3f} ± {best1['ap_std']:.3f}"
          f"  AUROC={best1['auroc']:.3f} ± {best1['auroc_std']:.3f}")
    print(f"  Class prevalence: {e1['prevalence'].iloc[0]:.1%}")

    # ------------------------------------------------------------------
    # Exp 02 — Feature set comparison
    # ------------------------------------------------------------------
    section("Exp 02  Feature Set Comparison (GBT, 5-fold CV)")
    e2 = pd.read_csv(BASE / "exp02_feature_comparison/feature_comparison.csv")
    print(e2[["feature_set", "n_features", "auroc", "auroc_std", "ap", "ap_std"]].to_string(index=False))
    best2 = e2.loc[e2["ap"].idxmax()]
    base2 = e2.iloc[0]
    gain = best2["ap"] / base2["ap"]
    print(f"\n  Best: {best2['feature_set'].strip()}  AP={best2['ap']:.3f} ± {best2['ap_std']:.3f}"
          f"  ({gain:.1f}x over biophysics-only)")
    print(f"  AUROC gain: {base2['auroc']:.3f} -> {best2['auroc']:.3f}"
          f"  (+{best2['auroc'] - base2['auroc']:.3f})")

    # ------------------------------------------------------------------
    # Exp 03 — AL vs random sampling
    # ------------------------------------------------------------------
    section("Exp 03  Active Learning vs Random (25 rounds × 10 queries, 5 trials)")
    e3 = pd.read_csv(BASE / "exp03_al_vs_random/summary.csv")
    print(e3[["strategy", "final_ap_mean", "final_ap_std",
              "final_rec_mean", "final_rec_std", "aulc"]].to_string(index=False))
    best3  = e3.loc[e3["final_rec_mean"].idxmax()]
    rand3  = e3[e3["strategy"] == "random"].iloc[0]
    recall_lift = best3["final_rec_mean"] - rand3["final_rec_mean"]
    print(f"\n  Best recall: {best3['strategy']}  recall={best3['final_rec_mean']:.3f}"
          f"  (+{recall_lift:.3f} vs random)")
    print(f"  Best AULC:   {e3.loc[e3['aulc'].idxmax(), 'strategy']}"
          f"  AULC={e3['aulc'].max():.2f}")

    # ------------------------------------------------------------------
    # Exp 04 — Hybrid acquisition strategies
    # ------------------------------------------------------------------
    section("Exp 04  Hybrid Acquisition Strategies (25 rounds × 10 queries, 5 trials)")
    e4 = pd.read_csv(BASE / "exp04_hybrid/strategy_summary.csv")
    print(e4[["strategy", "final_ap_mean", "final_ap_std",
              "final_rec_mean", "final_rec_std", "aulc"]].to_string(index=False))
    best4  = e4.loc[e4["final_rec_mean"].idxmax()]
    rand4  = e4[e4["strategy"] == "random"].iloc[0]
    rec_lift4 = best4["final_rec_mean"] - rand4["final_rec_mean"]
    print(f"\n  Best recall: {best4['strategy']}  recall={best4['final_rec_mean']:.3f}"
          f"  (+{rec_lift4:.3f} vs random)")

    # ------------------------------------------------------------------
    # Consolidated summary table
    # ------------------------------------------------------------------
    section("Summary — Key Finding per Experiment")
    rows = [
        ("Exp01 GBT baseline",
         "AP (5-fold CV)",
         f"{best1['ap']:.3f} ± {best1['ap_std']:.3f}"),
        ("Exp02 +GO/ESM-2 features",
         "AP (5-fold CV)",
         f"{best2['ap']:.3f}  ({gain:.1f}x over biophysics)"),
        ("Exp03 UCB strategy",
         "Recall @ 25 rounds",
         f"{e3.loc[e3['final_rec_mean'].idxmax(), 'final_rec_mean']:.3f}"
         f"  (+{recall_lift:.3f} vs random)"),
        ("Exp04 neural_score",
         "Recall @ 25 rounds",
         f"{best4['final_rec_mean']:.3f}"
         f"  (+{rec_lift4:.3f} vs random)"),
    ]
    hdr = f"{'Experiment':<28} {'Metric':<26} {'Value'}"
    print(hdr)
    print("-" * len(hdr))
    for exp, metric, val in rows:
        print(f"{exp:<28} {metric:<26} {val}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
