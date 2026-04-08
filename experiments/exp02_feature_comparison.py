"""
Experiment 2: Feature Set Comparison
======================================
Goal: Identify which feature representation best predicts neural relevance,
      so we know what to feed the active learning model in experiments 3–5.

Three feature sets (cumulative):
  A — Protein biophysics   (length, MW, charge, hydrophobicity, secondary structure)
  B — + Sequence composition  (21 amino-acid frequencies)
  C — + Evolutionary conservation  (human ortholog alignment score)

Model: Random Forest (class-weighted) — interpretable, robust to scale.

Metrics: AUROC | F1 | Average Precision
         + Feature importance table per set

Run from project root:
    python experiments/exp02_feature_comparison.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

# -- Configuration -------------------------------------------------------------
DATA_PATH = Path("data/processed/enriched_genomic_features.csv")
RESULTS_DIR = Path("artifacts/exp02_feature_comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5

NEURAL_GENE_IDS: dict[str, str] = {
    "WBGene00003734": "nlg-1",
    "WBGene00003816": "nrx-1",
    "WBGene00006745": "unc-13",
    "WBGene00004354": "ric-4",
    "WBGene00004944": "snb-1",
    "WBGene00001183": "eat-4",
    "WBGene00001617": "glr-1",
    "WBGene00003681": "nmr-1",
    "WBGene00000491": "cha-1",
    "WBGene00003152": "mec-4",
    "WBGene00006261": "tax-4",
    "WBGene00006259": "tax-2",
    "WBGene00001445": "flp-1",
    "WBGene00019756": "shn-1",
}

SET_A = [
    "length", "molecular_weight", "aromaticity", "instability_index",
    "isoelectric_point", "gravy", "helix_fraction", "turn_fraction", "sheet_fraction",
]


# -- Data ----------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    df = pd.read_csv(DATA_PATH)
    df["is_neural"] = df["common_name"].isin(NEURAL_GENE_IDS).astype(int)

    aa_cols = sorted(c for c in df.columns if c.startswith("percent_"))
    set_a = [c for c in SET_A if c in df.columns]
    set_b = set_a + aa_cols
    set_c = set_b + (["human_alignment_score"] if "human_alignment_score" in df.columns else [])

    agg = {c: "mean" for c in set_c}
    agg["is_neural"] = "max"
    gene_df = df.groupby("common_name").agg(agg).reset_index()

    feature_sets = {"A_biophysics": set_a, "B_+aa_composition": set_b, "C_+conservation": set_c}
    return gene_df, gene_df["is_neural"], feature_sets


# -- Evaluation ----------------------------------------------------------------

def evaluate_feature_set(
    gene_df: pd.DataFrame,
    y: pd.Series,
    feat_cols: list[str],
    label: str,
) -> dict:
    X = gene_df[feat_cols].fillna(0.0)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=-1)

    scores = cross_validate(
        model, X, y, cv=cv,
        scoring={"roc_auc": "roc_auc", "f1": "f1", "average_precision": "average_precision"},
    )

    result = {"FeatureSet": label, "n_features": len(feat_cols)}
    for metric in ("roc_auc", "f1", "average_precision"):
        vals = scores[f"test_{metric}"]
        result[metric] = vals.mean()
        result[f"{metric}_std"] = vals.std()

    print(
        f"  [{label:25s}] features={len(feat_cols):3d}  "
        f"AUROC {result['roc_auc']:.3f}+/-{result['roc_auc_std']:.3f}  "
        f"F1 {result['f1']:.3f}+/-{result['f1_std']:.3f}  "
        f"AP {result['average_precision']:.3f}+/-{result['average_precision_std']:.3f}"
    )

    # Feature importances (retrain on full data for stability)
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    imp.to_csv(RESULTS_DIR / f"importance_{label}.csv")

    return result


# -- Plots ---------------------------------------------------------------------

def plot_comparison(results_df: pd.DataFrame) -> None:
    metrics = ["roc_auc", "f1", "average_precision"]
    labels  = ["AUROC", "F1", "Avg Precision"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Exp 2 — Feature Set Comparison (Random Forest, 5-fold CV)",
                 fontsize=13, fontweight="bold")

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        vals = results_df[metric].values
        errs = results_df[f"{metric}_std"].values
        x_pos = np.arange(len(results_df))
        ax.bar(x_pos, vals, yerr=errs, capsize=5, color=color, alpha=0.85, width=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(results_df["FeatureSet"], rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.set_title(label)
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.text(i, v + e + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    out = RESULTS_DIR / "exp02_feature_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot -> {out}")
    plt.close()


def plot_top_importances(feature_sets: dict[str, list[str]], gene_df: pd.DataFrame,
                         y: pd.Series) -> None:
    """Plot top-10 feature importances side-by-side for each set."""
    fig, axes = plt.subplots(1, len(feature_sets), figsize=(5 * len(feature_sets), 5))
    fig.suptitle("Exp 2 — Top Feature Importances per Set", fontsize=12, fontweight="bold")

    for ax, (label, feats) in zip(axes, feature_sets.items()):
        X = gene_df[feats].fillna(0.0)
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X, y)
        imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False).head(10)
        imp.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"Set {label}")
        ax.invert_yaxis()
        ax.set_xlabel("Importance")

    plt.tight_layout()
    out = RESULTS_DIR / "exp02_importances.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot -> {out}")
    plt.close()


# -- Main ----------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Experiment 2: Feature Set Comparison")
    print("=" * 60)

    print(f"\nLoading {DATA_PATH} ...")
    gene_df, y, feature_sets = load_data()
    n_pos = int(y.sum())
    print(f"  Genes: {len(gene_df):,}  |  Neural-relevant: {n_pos} ({100 * y.mean():.3f}%)")

    if n_pos < 2:
        sys.exit("ERROR: Too few positives — check NEURAL_GENE_IDS.")

    print(f"\nEvaluating feature sets ({CV_FOLDS}-fold CV, RandomForest) ...")
    records = []
    for label, feats in feature_sets.items():
        records.append(evaluate_feature_set(gene_df, y, feats, label))

    results_df = pd.DataFrame(records)
    results_df.to_csv(RESULTS_DIR / "comparison_results.csv", index=False)

    print("\nSaving plots ...")
    plot_comparison(results_df)
    plot_top_importances(feature_sets, gene_df, y)

    print("\n" + "-" * 60)
    print(results_df[["FeatureSet", "n_features", "roc_auc", "f1", "average_precision"]].to_string(index=False))
    print("-" * 60)
    print(f"\nOutputs -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
