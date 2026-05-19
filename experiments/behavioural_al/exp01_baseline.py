# Exp01 - Baseline Model Comparison
# Cross-validate 3 classifiers to see how much predictive signal exists before any AL
# TODO: try different thresholds

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data_loader import load_gene_level_data

try:
    from pub_style import apply_pub_style
    apply_pub_style()
except ImportError:
    pass

os.makedirs("artifacts/behavioural_al/exp01_baseline", exist_ok=True)
RESULTS_DIR = Path("artifacts/behavioural_al/exp01_baseline")

TARGET_COL = "neural_behaviour_count"
THRESHOLD = 1   # gene counts as positive if it has >= 1 neural behaviour annotation
CV_FOLDS = 5
RANDOM_STATE = 42


def evaluate_model(name, model, X, y):
    """Stratified k-fold CV, returns mean AUROC and average precision."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scaler = StandardScaler()

    aurocs = []
    aps = []

    for train_idx, val_idx in skf.split(X, y):
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        model.fit(X_train, y[train_idx])
        proba = model.predict_proba(X_val)[:, 1]

        aurocs.append(roc_auc_score(y[val_idx], proba))
        aps.append(average_precision_score(y[val_idx], proba))

    return {
        "model": name,
        "auroc": np.mean(aurocs),
        "auroc_std": np.std(aurocs),
        "ap": np.mean(aps),
        "ap_std": np.std(aps),
    }


def main():
    print("=" * 60)
    print("Exp01 - Baseline Model Comparison")
    print(f"  Target: {TARGET_COL} >= {THRESHOLD}  |  CV folds: {CV_FOLDS}")
    print("=" * 60)

    # load data - not using ESM2/GO here to keep it simple
    gene_df, feat_cols = load_gene_level_data(
        target_col=TARGET_COL, include_esm2=False, include_go=False
    )

    gene_df["label"] = (gene_df[TARGET_COL] >= THRESHOLD).astype(int)

    # only confirmed genes - unknowns have no reliable label
    confirmed = gene_df[gene_df["has_annotation"] > 0].reset_index(drop=True)
    X = confirmed[feat_cols].values.astype(float)
    y = confirmed["label"].values.astype(int)

    n_pos = int(y.sum())
    print(f"  Confirmed genes : {len(confirmed):,}")
    print(f"  Positives       : {n_pos:,} ({100*y.mean():.1f}%)")
    print(f"  Features        : {len(feat_cols)}")
    print()

    # class_weight='balanced' helps with the ~12% positive rate
    models = {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }

    rows = []
    for name, model in models.items():
        print(f"  Evaluating {name} ...")
        result = evaluate_model(name, model, X, y)
        rows.append(result)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "cv_results.csv", index=False)

    # --- plot results ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"Exp01 - Baseline Models  |  {TARGET_COL} >= {THRESHOLD}  |  "
        f"{n_pos:,} positives ({100*y.mean():.1f}%)",
        fontweight="bold",
    )

    # short names for x-axis
    short_names = {"LogisticRegression": "LR", "RandomForest": "RF", "GradientBoosting": "GBT"}
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    labels = [short_names[r] for r in results["model"]]

    # AUROC bar chart
    bars = axes[0].bar(labels, results["auroc"], yerr=results["auroc_std"],
                       color=colors, capsize=5, alpha=0.85)
    axes[0].set_title("AUROC")
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.5, 1.0)
    for bar, v, e in zip(bars, results["auroc"], results["auroc_std"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + e + 0.01,
                     f"{v:.3f}", ha="center", fontsize=9)

    # Average Precision bar chart
    bars2 = axes[1].bar(labels, results["ap"], yerr=results["ap_std"],
                        color=colors, capsize=5, alpha=0.85)
    axes[1].set_title("Average Precision")
    axes[1].set_ylabel("AP")
    for bar, v, e in zip(bars2, results["ap"], results["ap_std"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + e + 0.01,
                     f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "baseline_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    print()
    print(results[["model", "auroc", "auroc_std", "ap", "ap_std"]].to_string(index=False))
    print(f"\nOutputs -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
