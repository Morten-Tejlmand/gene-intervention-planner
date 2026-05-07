"""
Experiment 1: Baseline Neural-Relevance Prediction
====================================================
Goal: Prove that gene features can predict neural-function relevance using
      simple baseline models. This validates that there is signal in the data
      to support active learning.

Models:  Logistic Regression | Random Forest | Gradient Boosting
Metrics: AUROC | F1 | Average Precision | Precision@K

Run from project root:
    python experiments/exp01_baseline_neural_relevance.py
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -- Configuration -------------------------------------------------------------
DATA_PATH = Path("data/processed/enriched_v2_features.csv")
RESULTS_DIR = Path("artifacts/exp01_baseline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5

# WBGene IDs for genes with well-established neural / synaptic function in C. elegans.
# These form the positive class. Extend this dict as new evidence emerges.
NEURAL_GENE_IDS: dict[str, str] = {
    # Synaptic adhesion / scaffolding
    "WBGene00003734": "nlg-1",
    "WBGene00003816": "nrx-1",
    "WBGene00019756": "shn-1",
    # Vesicle priming / fusion / release
    "WBGene00006745": "unc-13",
    "WBGene00004354": "ric-4",
    "WBGene00004944": "snb-1",
    "WBGene00006757": "unc-18",
    "WBGene00006798": "unc-64",
    "WBGene00006364": "syd-2",
    "WBGene00000086": "aex-3",
    "WBGene00006767": "unc-31",
    # Cholinergic
    "WBGene00000491": "cha-1",
    "WBGene00006756": "unc-17",
    "WBGene00000501": "cho-1",
    "WBGene00006765": "unc-29",
    "WBGene00002974": "lev-1",
    "WBGene00000042": "acr-2",
    "WBGene00000043": "acr-3",
    # GABAergic
    "WBGene00006762": "unc-25",
    "WBGene00006783": "unc-47",
    "WBGene00006784": "unc-49",
    "WBGene00001373": "exp-1",
    "WBGene00012915": "lgc-35",
    # Glutamatergic
    "WBGene00001183": "eat-4",
    "WBGene00001617": "glr-1",
    "WBGene00003681": "nmr-1",
    "WBGene00001613": "glr-2",
    "WBGene00001615": "glr-4",
    "WBGene00001616": "glr-5",
    # Dopaminergic
    "WBGene00000296": "cat-2",
    "WBGene00000934": "dat-1",
    "WBGene00001052": "dop-1",
    "WBGene00001053": "dop-2",
    "WBGene00020506": "dop-3",
    "WBGene00000295": "cat-1",
    # Serotonergic
    "WBGene00006600": "tph-1",
    "WBGene00003387": "mod-5",
    "WBGene00004776": "ser-1",
    "WBGene00004779": "ser-4",
    "WBGene00004780": "ser-7",
    # Mechanosensory / sensory
    "WBGene00003152": "mec-4",
    "WBGene00003174": "mec-10",
    "WBGene00003166": "mec-2",
    "WBGene00003167": "mec-3",
    "WBGene00003170": "mec-6",
    "WBGene00003889": "osm-9",
    "WBGene00003839": "ocr-2",
    "WBGene00006616": "trp-4",
    "WBGene00006261": "tax-4",
    "WBGene00006259": "tax-2",
    "WBGene00006527": "tax-6",
    # Neuropeptide
    "WBGene00001445": "flp-1",
    "WBGene00001172": "egl-3",
    "WBGene00001189": "egl-21",
    # Axon development
    "WBGene00004457": "rpm-1",
    "WBGene00001008": "dlk-1",
    # Neural transcription factors
    "WBGene00006818": "unc-86",
    "WBGene00006654": "ttx-3",
    "WBGene00000435": "ceh-10",
    # Other neural
    "WBGene00000149": "apl-1",
}

BIOPHYS_FEATURES = [
    "length",
    "molecular_weight",
    "aromaticity",
    "instability_index",
    "isoelectric_point",
    "gravy",
    "helix_fraction",
    "turn_fraction",
    "sheet_fraction",
]

FUNCTIONAL_FEATURES = [
    "annot_count",
    "unique_pheno_count",
    "positive_annot_count",
    "positive_annot_rate",
    "mean_evidence_weight",
    "has_annotation",
    "reference_count",
    "neural_pheno_overlap",
    "feature_ortholog_count",
    "feature_human_ortholog_count",
    "feature_max_query_identity",
    "vertebrate_ortholog_count",
    "ortholog_species_count",
]


# -- Data ----------------------------------------------------------------------


def load_gene_level_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    """Load protein features, aggregate isoforms to gene level, apply neural labels."""
    df = pd.read_csv(DATA_PATH)
    df["is_neural"] = df["common_name"].isin(NEURAL_GENE_IDS).astype(int)

    aa_cols = sorted(c for c in df.columns if c.startswith("percent_"))
    feat_cols = [
        c
        for c in BIOPHYS_FEATURES
        + ["human_alignment_score"]
        + aa_cols
        + FUNCTIONAL_FEATURES
        if c in df.columns
    ]

    agg = {c: "mean" for c in feat_cols}
    agg["is_neural"] = "max"  # gene is neural if ANY isoform matched

    gene_df = df.groupby("common_name").agg(agg).reset_index()
    gene_df["gene_name"] = (
        gene_df["common_name"].map(NEURAL_GENE_IDS).fillna(gene_df["common_name"])
    )

    X = gene_df[feat_cols].fillna(0.0)
    y = gene_df["is_neural"]
    return gene_df, X, y, feat_cols


# -- Models --------------------------------------------------------------------


def get_models() -> dict:
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 50) -> float:
    top_k = np.argsort(y_score)[::-1][:k]
    return float(y_true[top_k].mean())


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    records: list[dict] = []

    for name, model in get_models().items():
        print(f"  [{name}]")
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "roc_auc": "roc_auc",
                "f1": "f1",
                "average_precision": "average_precision",
            },
        )
        r: dict = {"Model": name}
        for metric in ("roc_auc", "f1", "average_precision"):
            vals = scores[f"test_{metric}"]
            r[metric] = vals.mean()
            r[f"{metric}_std"] = vals.std()
        records.append(r)
        print(
            f"    AUROC {r['roc_auc']:.3f}+/-{r['roc_auc_std']:.3f}  "
            f"F1 {r['f1']:.3f}+/-{r['f1_std']:.3f}  "
            f"AP {r['average_precision']:.3f}+/-{r['average_precision_std']:.3f}"
        )

    return pd.DataFrame(records).set_index("Model")


def compute_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )


# -- Plots ---------------------------------------------------------------------


def plot_results(results_df: pd.DataFrame, importance: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Exp 1 — Baseline Neural-Relevance Prediction", fontsize=13, fontweight="bold"
    )

    metrics = ["roc_auc", "f1", "average_precision"]
    err = results_df[[f"{m}_std" for m in metrics]].values.T
    results_df[metrics].rename(
        columns={"roc_auc": "AUROC", "f1": "F1", "average_precision": "Avg Precision"}
    ).plot(kind="bar", ax=axes[0], yerr=err, capsize=4, colormap="tab10")
    axes[0].set_title("5-Fold CV Performance")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(loc="upper right")

    importance.head(20).plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title("Top 20 Feature Importances (RandomForest)")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    out = RESULTS_DIR / "exp01_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot -> {out}")
    plt.close()


# -- Main ----------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Experiment 1: Baseline Neural-Relevance Prediction")
    print("=" * 60)

    print(f"\nLoading {DATA_PATH} ...")
    gene_df, X, y, feat_cols = load_gene_level_data()
    n_pos = int(y.sum())
    print(
        f"  Genes: {len(gene_df):,}  |  Neural-relevant: {n_pos} ({100 * y.mean():.3f}%)"
    )
    print(f"  Features: {X.shape[1]}")
    print(f"  Neural genes found: {gene_df[y == 1]['gene_name'].tolist()}")

    if n_pos < 2:
        sys.exit("ERROR: Too few positives — check NEURAL_GENE_IDS and data file.")

    print(f"\nRunning {CV_FOLDS}-fold stratified cross-validation ...")
    results_df = evaluate_models(X, y)

    results_df.to_csv(RESULTS_DIR / "cv_results.csv")

    print("\nComputing feature importance (RandomForest) ...")
    importance = compute_feature_importance(X, y)
    importance.to_csv(RESULTS_DIR / "feature_importance.csv")
    print(f"  Top 5: {', '.join(importance.head(5).index.tolist())}")

    print("\nSaving plots ...")
    plot_results(results_df, importance)

    print("\n" + "-" * 60)
    print(
        results_df[["roc_auc", "f1", "average_precision"]]
        .rename(
            columns={
                "roc_auc": "AUROC",
                "f1": "F1",
                "average_precision": "AvgPrecision",
            }
        )
        .to_string()
    )
    print("-" * 60)
    print(f"\nOutputs -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
