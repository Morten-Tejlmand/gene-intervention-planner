"""
Experiment 3: Active Learning vs Random Selection
===================================================
Core question: Does model-guided gene selection beat random under a limited
experiment budget?

Simulation protocol:
  - Pool:    all genes, with 12 known neural genes as the "oracle truth"
  - Seed:    8 neural genes + 40 random negatives (labeled at start)
  - Test:    4 held-out neural genes + 200 random negatives (never queried)
  - Rounds:  20  |  Queries per round: 15
  - Trials:  10 independent random seeds for error bars

Acquisition functions compared:
  random       — select uniformly at random from the unlabeled pool
  uncertainty  — margin sampling: select genes where model is least certain
                 score = 1 - |P(positive) - P(negative)|

Metrics:
  AUROC on fixed test set after each round
  Cumulative neural-gene recall in top-K of model ranking

Run from project root:
    python experiments/exp03_al_vs_random.py
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# -- Configuration -------------------------------------------------------------
DATA_PATH = Path("data/processed/enriched_genomic_features.csv")
RESULTS_DIR = Path("artifacts/exp03_al_vs_random")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_ROUNDS = 20
QUERY_SIZE = 15
N_SEED_POSITIVES = 8     # neural genes revealed at start
N_SEED_NEGATIVES = 40    # non-neural genes revealed at start
N_TEST_POSITIVES = 4     # held-out neural genes for evaluation
N_TEST_NEGATIVES = 200   # held-out non-neural genes for evaluation
N_TRIALS = 10
TOP_K = 50               # for recall-in-top-K metric
RANDOM_STATE = 42

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

FEATURE_COLS_BIOPHYS = [
    "length", "molecular_weight", "aromaticity", "instability_index",
    "isoelectric_point", "gravy", "helix_fraction", "turn_fraction", "sheet_fraction",
    "human_alignment_score",
]


# -- Data ----------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["is_neural"] = df["common_name"].isin(NEURAL_GENE_IDS).astype(int)

    aa_cols = sorted(c for c in df.columns if c.startswith("percent_"))
    feat_cols = [c for c in FEATURE_COLS_BIOPHYS + aa_cols if c in df.columns]

    agg = {c: "mean" for c in feat_cols}
    agg["is_neural"] = "max"
    gene_df = df.groupby("common_name").agg(agg).reset_index()
    gene_df["_feat_cols"] = None  # store col list separately
    gene_df.attrs["feat_cols"] = feat_cols
    return gene_df


# -- Model ---------------------------------------------------------------------

def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )


# -- Acquisition Functions -----------------------------------------------------

def acquire_random(pool_idx: np.ndarray, *, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(pool_idx, size=min(k, len(pool_idx)), replace=False)


def acquire_uncertainty(
    pool_idx: np.ndarray,
    model: RandomForestClassifier,
    X: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    proba = model.predict_proba(X[pool_idx])
    # Margin sampling: lower margin -> higher uncertainty
    if proba.shape[1] == 2:
        margin = np.abs(proba[:, 1] - proba[:, 0])
    else:
        sorted_proba = np.sort(proba, axis=1)
        margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    top = np.argsort(margin)[:k]
    return pool_idx[top]


# -- Simulation ----------------------------------------------------------------

def run_trial(
    gene_df: pd.DataFrame,
    feat_cols: list[str],
    strategy: str,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    """Run one AL trial; return AUROC and recall-in-top-K per round."""
    all_idx = np.arange(len(gene_df))
    y = gene_df["is_neural"].values
    X_raw = gene_df[feat_cols].fillna(0.0).values

    # Normalize once
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) < N_SEED_POSITIVES + N_TEST_POSITIVES:
        raise ValueError(
            f"Not enough positives: need {N_SEED_POSITIVES + N_TEST_POSITIVES}, got {len(pos_idx)}"
        )

    # Split positives into seed / test
    rng.shuffle(pos_idx)
    seed_pos = pos_idx[:N_SEED_POSITIVES]
    test_pos = pos_idx[N_SEED_POSITIVES: N_SEED_POSITIVES + N_TEST_POSITIVES]

    # Split negatives into seed / test / pool
    rng.shuffle(neg_idx)
    seed_neg = neg_idx[:N_SEED_NEGATIVES]
    test_neg = neg_idx[N_SEED_NEGATIVES: N_SEED_NEGATIVES + N_TEST_NEGATIVES]
    pool_neg = neg_idx[N_SEED_NEGATIVES + N_TEST_NEGATIVES:]

    test_idx = np.concatenate([test_pos, test_neg])
    labeled_idx = np.concatenate([seed_pos, seed_neg])
    pool_idx = np.concatenate([pool_neg])  # remaining negatives; positives enter via oracle

    # Positives not yet revealed (oracle will reveal them when selected, or they stay hidden)
    hidden_pos = pos_idx[N_SEED_POSITIVES + N_TEST_POSITIVES:]

    auroc_curve: list[float] = []
    recall_curve: list[float] = []

    for _round in range(N_ROUNDS + 1):
        # Eval
        model = make_model()
        if len(np.unique(y[labeled_idx])) < 2:
            auroc_curve.append(0.5)
            recall_curve.append(0.0)
        else:
            model.fit(X[labeled_idx], y[labeled_idx])
            y_prob = model.predict_proba(X[test_idx])[:, -1]
            auroc = roc_auc_score(y[test_idx], y_prob)
            auroc_curve.append(float(auroc))

            # Recall in top-K over the full unlabeled pool + hidden positives
            unlab_idx = np.concatenate([pool_idx, hidden_pos])
            if len(unlab_idx) > 0:
                scores = model.predict_proba(X[unlab_idx])[:, -1]
                top_k = np.argsort(scores)[::-1][: min(TOP_K, len(unlab_idx))]
                n_pos_found = int(y[unlab_idx[top_k]].sum())
                recall_curve.append(n_pos_found / max(1, len(hidden_pos) + N_TEST_POSITIVES - len(test_pos)))
            else:
                recall_curve.append(0.0)

        if _round == N_ROUNDS:
            break

        # Acquire
        if strategy == "random":
            chosen = acquire_random(pool_idx, k=QUERY_SIZE, rng=rng)
        else:  # uncertainty
            if len(np.unique(y[labeled_idx])) < 2:
                chosen = acquire_random(pool_idx, k=QUERY_SIZE, rng=rng)
            else:
                chosen = acquire_uncertainty(pool_idx, model, X, k=QUERY_SIZE)

        # Reveal labels and update sets
        labeled_idx = np.concatenate([labeled_idx, chosen])
        pool_idx = np.setdiff1d(pool_idx, chosen)

    return {"auroc": auroc_curve, "recall_top_k": recall_curve}


def run_strategy(gene_df: pd.DataFrame, feat_cols: list[str], strategy: str) -> dict:
    print(f"  [{strategy}] running {N_TRIALS} trials ...")
    all_auroc, all_recall = [], []
    for t in range(N_TRIALS):
        rng = np.random.default_rng(RANDOM_STATE + t)
        trial = run_trial(gene_df, feat_cols, strategy, rng)
        all_auroc.append(trial["auroc"])
        all_recall.append(trial["recall_top_k"])

    auroc = np.array(all_auroc)
    recall = np.array(all_recall)
    return {
        "auroc_mean": auroc.mean(0),
        "auroc_std": auroc.std(0),
        "recall_mean": recall.mean(0),
        "recall_std": recall.std(0),
    }


# -- Plots ---------------------------------------------------------------------

def plot_curves(results: dict[str, dict]) -> None:
    rounds = np.arange(N_ROUNDS + 1)
    labeled_counts = (N_SEED_POSITIVES + N_SEED_NEGATIVES) + rounds * QUERY_SIZE
    colors = {"random": "gray", "uncertainty": "steelblue"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Exp 3 — Active Learning vs Random  "
        f"({N_TRIALS} trials, query={QUERY_SIZE}/round)",
        fontsize=12, fontweight="bold",
    )

    for strategy, res in results.items():
        c = colors.get(strategy, "orange")
        axes[0].plot(labeled_counts, res["auroc_mean"], label=strategy, color=c, lw=2)
        axes[0].fill_between(
            labeled_counts,
            res["auroc_mean"] - res["auroc_std"],
            res["auroc_mean"] + res["auroc_std"],
            alpha=0.2, color=c,
        )
        axes[1].plot(labeled_counts, res["recall_mean"], label=strategy, color=c, lw=2)
        axes[1].fill_between(
            labeled_counts,
            res["recall_mean"] - res["recall_std"],
            res["recall_mean"] + res["recall_std"],
            alpha=0.2, color=c,
        )

    axes[0].set_title("AUROC on held-out test set")
    axes[0].set_xlabel("Labeled genes (budget spent)")
    axes[0].set_ylabel("AUROC")
    axes[0].legend()
    axes[0].set_ylim(0.4, 1.0)

    axes[1].set_title(f"Neural-gene recall in top-{TOP_K} model ranking")
    axes[1].set_xlabel("Labeled genes (budget spent)")
    axes[1].set_ylabel(f"Recall@{TOP_K}")
    axes[1].legend()

    plt.tight_layout()
    out = RESULTS_DIR / "exp03_learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot -> {out}")
    plt.close()


# -- Main ----------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Experiment 3: Active Learning vs Random Selection")
    print("=" * 60)
    print(f"  Rounds={N_ROUNDS}  Query={QUERY_SIZE}/round  Trials={N_TRIALS}")

    print(f"\nLoading {DATA_PATH} ...")
    gene_df = load_data()
    feat_cols = gene_df.attrs["feat_cols"]
    del gene_df["_feat_cols"]
    n_pos = int(gene_df["is_neural"].sum())
    print(f"  Genes: {len(gene_df):,}  |  Neural-relevant: {n_pos}")

    if n_pos < N_SEED_POSITIVES + N_TEST_POSITIVES:
        sys.exit(
            f"ERROR: Need at least {N_SEED_POSITIVES + N_TEST_POSITIVES} positive genes, "
            f"found {n_pos}. Reduce N_SEED_POSITIVES or N_TEST_POSITIVES."
        )

    results: dict[str, dict] = {}
    for strategy in ("random", "uncertainty"):
        results[strategy] = run_strategy(gene_df, feat_cols, strategy)

    # Save summary
    summary_rows = []
    for strategy, res in results.items():
        summary_rows.append({
            "strategy": strategy,
            "final_auroc_mean": res["auroc_mean"][-1],
            "final_auroc_std": res["auroc_std"][-1],
            "final_recall_mean": res["recall_mean"][-1],
            "final_recall_std": res["recall_std"][-1],
            "auc_learning_curve": float(np.trapezoid(res["auroc_mean"])),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    # Save full curves
    for strategy, res in results.items():
        pd.DataFrame(res).to_csv(RESULTS_DIR / f"curves_{strategy}.csv", index=False)

    print("\nSaving plots ...")
    plot_curves(results)

    print("\n" + "-" * 60)
    print(summary.to_string(index=False))
    print("-" * 60)
    print(f"\nOutputs -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
