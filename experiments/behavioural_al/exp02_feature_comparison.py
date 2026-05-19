# Exp02 - Feature Set Comparison
# Test how much each feature group contributes to predicting neural-behavioural genes
# Using GBT since it performed best in exp01
#
# Feature sets (cumulative):
#   A = biophysical only (length, MW, etc.)
#   B = A + amino acid composition
#   C = B + evolutionary/ortholog features
#   D = C + GO term SVD + ESM-2 embeddings (if available)

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data_loader import BIOPHYS_FEATURES, ORTHOLOG_FEATURES, load_gene_level_data

try:
    from pub_style import apply_pub_style
    apply_pub_style()
except ImportError:
    pass

os.makedirs("artifacts/behavioural_al/exp02_feature_comparison", exist_ok=True)
RESULTS_DIR = Path("artifacts/behavioural_al/exp02_feature_comparison")

TARGET_COL = "neural_behaviour_count"
THRESHOLD = 1
CV_FOLDS = 5
RANDOM_STATE = 42


def cv_evaluate(cols, X, y):
    """5-fold stratified CV with GBT, returns mean AUROC and AP."""
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE
    )
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
        "n_features": len(cols),
        "auroc": np.mean(aurocs),
        "auroc_std": np.std(aurocs),
        "ap": np.mean(aps),
        "ap_std": np.std(aps),
    }


def main():
    print("=" * 60)
    print("Exp02 - Feature Set Comparison")
    print(f"  Target: {TARGET_COL} >= {THRESHOLD}  |  Classifier: GBT  |  CV: {CV_FOLDS}")
    print("=" * 60)

    # load everything including ESM2 and GO (needed for set D)
    gene_df, all_feat_cols = load_gene_level_data(
        target_col=TARGET_COL, include_esm2=True, include_go=True
    )
    gene_df["label"] = (gene_df[TARGET_COL] >= THRESHOLD).astype(int)

    # only confirmed genes
    confirmed = gene_df[gene_df["has_annotation"] > 0].reset_index(drop=True)
    y = confirmed["label"].values.astype(int)

    # build the four feature sets
    aa_cols = sorted([c for c in confirmed.columns if c.startswith("percent_")])
    biophys_cols = [c for c in BIOPHYS_FEATURES if c in confirmed.columns]
    if "human_alignment_score" in confirmed.columns:
        biophys_cols = biophys_cols + ["human_alignment_score"]
    ortholog_cols = [c for c in ORTHOLOG_FEATURES if c in confirmed.columns]
    esm2_cols = [c for c in all_feat_cols if c.startswith("esm2_")]
    go_cols = [c for c in all_feat_cols if c.startswith("go_svd_")]

    feature_sets = {
        "A — Biophysical": biophys_cols,
        "B — +AA composition": biophys_cols + aa_cols,
        "C — +Evolutionary": biophys_cols + aa_cols + ortholog_cols,
    }

    # only add set D if we have the rich features
    if esm2_cols or go_cols:
        feature_sets["D — +GO/ESM-2"] = biophys_cols + aa_cols + ortholog_cols + esm2_cols + go_cols
    else:
        print("  Note: ESM-2/GO files not found, skipping set D")

    print(f"  Confirmed genes : {len(confirmed):,}  |  Positives: {int(y.sum()):,} ({100*y.mean():.1f}%)")
    print()

    rows = []
    for set_name, cols in feature_sets.items():
        X = confirmed[cols].values.astype(float)
        print(f"  [{set_name}]  {len(cols)} features ...")
        res = cv_evaluate(cols, X, y)
        res["feature_set"] = set_name
        rows.append(res)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "feature_comparison.csv", index=False)

    # --- plot ---
    set_names = results["feature_set"].tolist()
    short_labels = [n.split(" — ")[0] for n in set_names]  # just "A", "B", "C", "D"
    colors = ["#9C27B0", "#2196F3", "#FF9800", "#4CAF50"][:len(set_names)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f"Exp02 - Feature Set Comparison  |  GBT 5-fold CV  |  target: {TARGET_COL} >= {THRESHOLD}",
        fontweight="bold",
    )

    for ax, col, err_col, title, ylim in [
        (axes[0], "auroc", "auroc_std", "AUROC", (0.5, 1.0)),
        (axes[1], "ap", "ap_std", "Average Precision", (0.0, None)),
    ]:
        bars = ax.bar(short_labels, results[col], yerr=results[err_col],
                      color=colors, capsize=5, alpha=0.85)
        ax.set(title=title, ylabel=title, ylim=ylim)
        for bar, v, e in zip(bars, results[col], results[err_col]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.01,
                    f"{v:.3f}", ha="center", fontsize=9)

    # add legend to second plot
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.85) for i in range(len(set_names))]
    axes[1].legend(handles, set_names, fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    print()
    print(results[["feature_set", "n_features", "auroc", "auroc_std", "ap", "ap_std"]].to_string(index=False))
    print(f"\nOutputs -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
