"""
Experiment 4: Hybrid Acquisition Strategy Comparison
======================================================
Goal: Compare acquisition strategies that go beyond pure uncertainty to
      incorporate biological relevance — avoiding informationally-rich but
      neurologically uninteresting genes.

Strategies:
  uncertainty    — margin sampling (same as Exp 3)
  neural_score   — select genes with highest predicted neural probability
                   (exploitation: mine model confidence directly)
  hybrid         — combined score:
                     score = alpha-uncertainty + beta-neural_prob + gamma-conservation
                   with alpha=0.4, beta=0.4, gamma=0.2 (tunable via HYBRID_WEIGHTS)

Metrics per round (10 trials):
  AUROC on fixed test set
  Proportion of neural genes recovered in queried set (enrichment)
  Top-K enrichment: neural genes found in top-50 model predictions

Run from project root:
    python experiments/exp04_hybrid_acquisition.py
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# -- Configuration -------------------------------------------------------------
DATA_PATH = Path("data/processed/enriched_genomic_features.csv")
RESULTS_DIR = Path("artifacts/exp04_hybrid_acquisition")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_ROUNDS = 20
QUERY_SIZE = 15
N_SEED_POSITIVES = 8
N_SEED_NEGATIVES = 40
N_TEST_POSITIVES = 4
N_TEST_NEGATIVES = 200
N_TRIALS = 10
TOP_K = 50
RANDOM_STATE = 42

# Hybrid score weights: [uncertainty, neural_prob, conservation]
HYBRID_WEIGHTS = (0.4, 0.4, 0.2)

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

STRATEGIES = ["uncertainty", "neural_score", "hybrid"]


# -- Data ----------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA_PATH)
    df["is_neural"] = df["common_name"].isin(NEURAL_GENE_IDS).astype(int)

    aa_cols = sorted(c for c in df.columns if c.startswith("percent_"))
    feat_cols = [c for c in FEATURE_COLS_BIOPHYS + aa_cols if c in df.columns]

    agg = {c: "mean" for c in feat_cols}
    agg["is_neural"] = "max"
    gene_df = df.groupby("common_name").agg(agg).reset_index()
    return gene_df, feat_cols


# -- Acquisition Functions -----------------------------------------------------

def _margin(proba: np.ndarray) -> np.ndarray:
    """Uncertainty: low margin -> high uncertainty score."""
    if proba.shape[1] == 2:
        margin = np.abs(proba[:, 1] - proba[:, 0])
    else:
        s = np.sort(proba, axis=1)
        margin = s[:, -1] - s[:, -2]
    return 1.0 - margin  # invert: high score = high uncertainty


def acquire(
    pool_idx: np.ndarray,
    model: RandomForestClassifier | None,
    X: np.ndarray,
    conservation: np.ndarray,
    strategy: str,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if model is None or strategy == "random":
        return rng.choice(pool_idx, size=min(k, len(pool_idx)), replace=False)

    proba = model.predict_proba(X[pool_idx])
    neural_prob = proba[:, -1]  # P(neural=1)
    uncertainty = _margin(proba)

    if strategy == "uncertainty":
        scores = uncertainty
    elif strategy == "neural_score":
        scores = neural_prob
    else:  # hybrid
        w_u, w_n, w_c = HYBRID_WEIGHTS
        # Normalise each component to [0, 1] across pool
        def _norm(v: np.ndarray) -> np.ndarray:
            rng_ = v.max() - v.min()
            return (v - v.min()) / rng_ if rng_ > 0 else np.zeros_like(v)

        cons = conservation[pool_idx]
        scores = w_u * _norm(uncertainty) + w_n * _norm(neural_prob) + w_c * _norm(cons)

    top = np.argsort(scores)[::-1][: min(k, len(pool_idx))]
    return pool_idx[top]


# -- Simulation ----------------------------------------------------------------

def run_trial(
    gene_df: pd.DataFrame,
    feat_cols: list[str],
    conservation: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
) -> dict[str, list]:
    y = gene_df["is_neural"].values
    X_raw = gene_df[feat_cols].fillna(0.0).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rng.shuffle(pos_idx)
    seed_pos = pos_idx[:N_SEED_POSITIVES]
    test_pos = pos_idx[N_SEED_POSITIVES: N_SEED_POSITIVES + N_TEST_POSITIVES]
    hidden_pos = pos_idx[N_SEED_POSITIVES + N_TEST_POSITIVES:]

    rng.shuffle(neg_idx)
    seed_neg = neg_idx[:N_SEED_NEGATIVES]
    test_neg = neg_idx[N_SEED_NEGATIVES: N_SEED_NEGATIVES + N_TEST_NEGATIVES]
    pool_neg = neg_idx[N_SEED_NEGATIVES + N_TEST_NEGATIVES:]

    test_idx = np.concatenate([test_pos, test_neg])
    labeled_idx = np.concatenate([seed_pos, seed_neg])
    pool_idx = pool_neg.copy()

    auroc_curve: list[float] = []
    enrichment_curve: list[float] = []  # proportion of queried genes that are neural
    recall_curve: list[float] = []

    queried_neural = 0
    total_queried = 0

    for _round in range(N_ROUNDS + 1):
        model = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=RANDOM_STATE, n_jobs=-1)

        if len(np.unique(y[labeled_idx])) < 2:
            auroc_curve.append(0.5)
            enrichment_curve.append(0.0)
            recall_curve.append(0.0)
        else:
            model.fit(X[labeled_idx], y[labeled_idx])
            y_prob = model.predict_proba(X[test_idx])[:, -1]
            auroc_curve.append(float(roc_auc_score(y[test_idx], y_prob)))

            enrichment_curve.append(queried_neural / max(1, total_queried))

            unlab = np.concatenate([pool_idx, hidden_pos])
            if len(unlab) > 0:
                scores = model.predict_proba(X[unlab])[:, -1]
                top_k_idx = np.argsort(scores)[::-1][: min(TOP_K, len(unlab))]
                n_found = int(y[unlab[top_k_idx]].sum())
                denom = len(hidden_pos) + len(test_pos)
                recall_curve.append(n_found / max(1, denom))
            else:
                recall_curve.append(1.0)

        if _round == N_ROUNDS:
            break

        # Acquire
        if len(np.unique(y[labeled_idx])) < 2:
            chosen = acquire(pool_idx, None, X, conservation, "random", QUERY_SIZE, rng)
        else:
            chosen = acquire(pool_idx, model, X, conservation, strategy, QUERY_SIZE, rng)

        queried_neural += int(y[chosen].sum())
        total_queried += len(chosen)

        labeled_idx = np.concatenate([labeled_idx, chosen])
        pool_idx = np.setdiff1d(pool_idx, chosen)

    return {"auroc": auroc_curve, "enrichment": enrichment_curve, "recall": recall_curve}


def run_strategy(gene_df: pd.DataFrame, feat_cols: list[str],
                 conservation: np.ndarray, strategy: str) -> dict:
    print(f"  [{strategy}] {N_TRIALS} trials ...")
    all_auroc, all_enrich, all_recall = [], [], []

    for t in range(N_TRIALS):
        rng = np.random.default_rng(RANDOM_STATE + t)
        trial = run_trial(gene_df, feat_cols, conservation, strategy, rng)
        all_auroc.append(trial["auroc"])
        all_enrich.append(trial["enrichment"])
        all_recall.append(trial["recall"])

    return {
        "auroc_mean": np.array(all_auroc).mean(0),
        "auroc_std": np.array(all_auroc).std(0),
        "enrich_mean": np.array(all_enrich).mean(0),
        "enrich_std": np.array(all_enrich).std(0),
        "recall_mean": np.array(all_recall).mean(0),
        "recall_std": np.array(all_recall).std(0),
    }


# -- Plots ---------------------------------------------------------------------

def plot_results(results: dict[str, dict]) -> None:
    rounds = np.arange(N_ROUNDS + 1)
    labeled_counts = (N_SEED_POSITIVES + N_SEED_NEGATIVES) + rounds * QUERY_SIZE
    colors = {"uncertainty": "steelblue", "neural_score": "darkorange", "hybrid": "seagreen"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Exp 4 — Hybrid Acquisition Strategy Comparison  "
        f"({N_TRIALS} trials, query={QUERY_SIZE}/round, weights={HYBRID_WEIGHTS})",
        fontsize=11, fontweight="bold",
    )

    for strategy, res in results.items():
        c = colors.get(strategy, "purple")
        kw = dict(label=strategy, color=c, lw=2)

        axes[0].plot(labeled_counts, res["auroc_mean"], **kw)
        axes[0].fill_between(labeled_counts,
                             res["auroc_mean"] - res["auroc_std"],
                             res["auroc_mean"] + res["auroc_std"],
                             alpha=0.15, color=c)

        axes[1].plot(labeled_counts, res["enrich_mean"], **kw)
        axes[1].fill_between(labeled_counts,
                             res["enrich_mean"] - res["enrich_std"],
                             res["enrich_mean"] + res["enrich_std"],
                             alpha=0.15, color=c)

        axes[2].plot(labeled_counts, res["recall_mean"], **kw)
        axes[2].fill_between(labeled_counts,
                             res["recall_mean"] - res["recall_std"],
                             res["recall_mean"] + res["recall_std"],
                             alpha=0.15, color=c)

    titles = [
        "AUROC on test set",
        "Neural enrichment in queried genes",
        f"Neural recall in top-{TOP_K} predictions",
    ]
    ylabels = ["AUROC", "Proportion neural (queried)", f"Recall@{TOP_K}"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title)
        ax.set_xlabel("Labeled genes (budget)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = RESULTS_DIR / "exp04_acquisition_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot -> {out}")
    plt.close()


def plot_final_bar(results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Exp 4 — Final-Round Summary", fontsize=12, fontweight="bold")

    metrics = [("auroc_mean", "auroc_std", "AUROC"),
               ("enrich_mean", "enrich_std", "Neural Enrichment"),
               ("recall_mean", "recall_std", f"Recall@{TOP_K}")]

    strategies = list(results.keys())
    colors = ["steelblue", "darkorange", "seagreen"]

    for ax, (m_key, s_key, label) in zip(axes, metrics):
        vals = [results[s][m_key][-1] for s in strategies]
        errs = [results[s][s_key][-1] for s in strategies]
        ax.bar(strategies, vals, yerr=errs, capsize=5,
               color=colors[:len(strategies)], alpha=0.85)
        ax.set_title(label)
        ax.set_ylim(0, max(vals) * 1.3 + 0.05)
        ax.tick_params(axis="x", rotation=15)
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.text(i, v + e + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    out = RESULTS_DIR / "exp04_final_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot -> {out}")
    plt.close()


# -- Main ----------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Experiment 4: Hybrid Acquisition Strategy Comparison")
    print("=" * 60)
    print(f"  Weights (uncertainty, neural_prob, conservation) = {HYBRID_WEIGHTS}")

    print(f"\nLoading {DATA_PATH} ...")
    gene_df, feat_cols = load_data()
    n_pos = int(gene_df["is_neural"].sum())
    print(f"  Genes: {len(gene_df):,}  |  Neural-relevant: {n_pos}")

    if n_pos < N_SEED_POSITIVES + N_TEST_POSITIVES:
        sys.exit(f"ERROR: Need {N_SEED_POSITIVES + N_TEST_POSITIVES} positives, have {n_pos}.")

    # Conservation signal: normalised human alignment score
    raw_cons = gene_df["human_alignment_score"].fillna(0.0).values if "human_alignment_score" in gene_df.columns else np.zeros(len(gene_df))
    mms = MinMaxScaler()
    conservation = mms.fit_transform(raw_cons.reshape(-1, 1)).flatten()

    results: dict[str, dict] = {}
    for strategy in STRATEGIES:
        results[strategy] = run_strategy(gene_df, feat_cols, conservation, strategy)

    # Save
    summary_rows = []
    for strategy, res in results.items():
        summary_rows.append({
            "strategy": strategy,
            "final_auroc": res["auroc_mean"][-1],
            "final_auroc_std": res["auroc_std"][-1],
            "final_enrichment": res["enrich_mean"][-1],
            "final_recall": res["recall_mean"][-1],
            "auc_learning_curve": float(np.trapezoid(res["auroc_mean"])),
        })
        pd.DataFrame({k: v for k, v in res.items() if isinstance(v, np.ndarray)}).to_csv(
            RESULTS_DIR / f"curves_{strategy}.csv", index=False
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    print("\nSaving plots ...")
    plot_results(results)
    plot_final_bar(results)

    print("\n" + "-" * 60)
    print(summary.to_string(index=False))
    print("-" * 60)
    print(f"\nOutputs -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
