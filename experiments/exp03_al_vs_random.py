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
  qbc          — query-by-committee (3× LR with C=0.1/1/10):
                 select genes with highest prediction variance across the committee

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# -- Configuration -------------------------------------------------------------
DATA_PATH = Path("data/processed/enriched_v2_features.csv")
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
    # Synaptic adhesion / scaffolding
    "WBGene00003734": "nlg-1",   "WBGene00003816": "nrx-1",
    "WBGene00019756": "shn-1",
    # Vesicle priming / fusion / release
    "WBGene00006745": "unc-13",  "WBGene00004354": "ric-4",
    "WBGene00004944": "snb-1",   "WBGene00006757": "unc-18",
    "WBGene00006798": "unc-64",  "WBGene00006364": "syd-2",
    "WBGene00000086": "aex-3",   "WBGene00006767": "unc-31",
    # Cholinergic
    "WBGene00000491": "cha-1",   "WBGene00006756": "unc-17",
    "WBGene00000501": "cho-1",   "WBGene00006765": "unc-29",
    "WBGene00002974": "lev-1",   "WBGene00000042": "acr-2",
    "WBGene00000043": "acr-3",
    # GABAergic
    "WBGene00006762": "unc-25",  "WBGene00006783": "unc-47",
    "WBGene00006784": "unc-49",  "WBGene00001373": "exp-1",
    "WBGene00012915": "lgc-35",
    # Glutamatergic
    "WBGene00001183": "eat-4",   "WBGene00001617": "glr-1",
    "WBGene00003681": "nmr-1",   "WBGene00001613": "glr-2",
    "WBGene00001615": "glr-4",   "WBGene00001616": "glr-5",
    # Dopaminergic
    "WBGene00000296": "cat-2",   "WBGene00000934": "dat-1",
    "WBGene00001052": "dop-1",   "WBGene00001053": "dop-2",
    "WBGene00020506": "dop-3",   "WBGene00000295": "cat-1",
    # Serotonergic
    "WBGene00006600": "tph-1",   "WBGene00003387": "mod-5",
    "WBGene00004776": "ser-1",   "WBGene00004779": "ser-4",
    "WBGene00004780": "ser-7",
    # Mechanosensory / sensory
    "WBGene00003152": "mec-4",   "WBGene00003174": "mec-10",
    "WBGene00003166": "mec-2",   "WBGene00003167": "mec-3",
    "WBGene00003170": "mec-6",   "WBGene00003889": "osm-9",
    "WBGene00003839": "ocr-2",   "WBGene00006616": "trp-4",
    "WBGene00006261": "tax-4",   "WBGene00006259": "tax-2",
    "WBGene00006527": "tax-6",
    # Neuropeptide
    "WBGene00001445": "flp-1",   "WBGene00001172": "egl-3",
    "WBGene00001189": "egl-21",
    # Axon development
    "WBGene00004457": "rpm-1",   "WBGene00001008": "dlk-1",
    # Neural transcription factors
    "WBGene00006818": "unc-86",  "WBGene00006654": "ttx-3",
    "WBGene00000435": "ceh-10",
    # Other neural
    "WBGene00000149": "apl-1",
}

FEATURE_COLS_BIOPHYS = [
    "length", "molecular_weight", "aromaticity", "instability_index",
    "isoelectric_point", "gravy", "helix_fraction", "turn_fraction", "sheet_fraction",
    "human_alignment_score",
]

FUNCTIONAL_FEATURES = [
    "annot_count", "unique_pheno_count", "positive_annot_count",
    "positive_annot_rate", "mean_evidence_weight", "has_annotation",
    "reference_count", "neural_pheno_overlap",
    "feature_ortholog_count", "feature_human_ortholog_count",
    "feature_max_query_identity", "vertebrate_ortholog_count",
    "ortholog_species_count",
]


# -- Data ----------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Match on both WBGene IDs (keys) and gene symbols (values)
    neural_ids = set(NEURAL_GENE_IDS.keys()) | set(NEURAL_GENE_IDS.values())
    df["is_neural"] = df["common_name"].isin(neural_ids).astype(int)

    aa_cols = sorted(c for c in df.columns if c.startswith("percent_"))
    feat_cols = [c for c in FEATURE_COLS_BIOPHYS + aa_cols + FUNCTIONAL_FEATURES if c in df.columns]

    agg = {c: "mean" for c in feat_cols}
    agg["is_neural"] = "max"
    gene_df = df.groupby("common_name").agg(agg).reset_index()
    gene_df["_feat_cols"] = None  # store col list separately
    gene_df.attrs["feat_cols"] = feat_cols
    return gene_df


# -- Model ---------------------------------------------------------------------

def make_model() -> LogisticRegression:
    return LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced",
        random_state=RANDOM_STATE,
    )


# -- Acquisition Functions -----------------------------------------------------

def acquire_random(pool_idx: np.ndarray, *, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(pool_idx, size=min(k, len(pool_idx)), replace=False)


def acquire_uncertainty(
    pool_idx: np.ndarray,
    model,
    X: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    proba = model.predict_proba(X[pool_idx])
    if proba.shape[1] == 2:
        margin = np.abs(proba[:, 1] - proba[:, 0])
    else:
        sorted_proba = np.sort(proba, axis=1)
        margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    top = np.argsort(margin)[:k]
    return pool_idx[top]


def make_committee() -> list:
    # Three LR with different regularization strengths for diverse predictions
    return [
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                           random_state=RANDOM_STATE),
        LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced",
                           random_state=RANDOM_STATE),
        LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced",
                           random_state=RANDOM_STATE),
    ]


def acquire_qbc(
    pool_idx: np.ndarray,
    committee: list,
    X: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Select genes where committee members disagree most (variance of P(neural=1))."""
    probas = np.stack(
        [m.predict_proba(X[pool_idx])[:, -1] for m in committee], axis=1
    )
    disagreement = probas.var(axis=1)
    top = np.argsort(disagreement)[::-1][:k]
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

    # Hidden positives join the queryable pool — the oracle reveals their label when queried.
    # Previously they were sequestered, making every query a negative and recall never improve.
    hidden_pos = pos_idx[N_SEED_POSITIVES + N_TEST_POSITIVES:]
    pool_idx = np.concatenate([pool_neg, hidden_pos])
    n_total_hidden = len(hidden_pos)

    ap_curve: list[float] = []
    recall_curve: list[float] = []

    for _round in range(N_ROUNDS + 1):
        has_both = len(np.unique(y[labeled_idx])) >= 2

        # Train model(s)
        model = make_model()
        committee: list | None = None
        if has_both:
            if strategy == "qbc":
                committee = make_committee()
                for m in committee:
                    m.fit(X[labeled_idx], y[labeled_idx])
                model = committee[0]  # LR (C=1) used for eval
            else:
                model.fit(X[labeled_idx], y[labeled_idx])

        # Eval — AP is primary (meaningful at 0.29% imbalance)
        if not has_both:
            ap_curve.append(0.0)
            recall_curve.append(0.0)
        else:
            y_prob = model.predict_proba(X[test_idx])[:, -1]
            ap_curve.append(float(average_precision_score(y[test_idx], y_prob)))

            # Recall: how many of the originally-hidden positives have been labeled?
            n_discovered = int(y[labeled_idx].sum()) - N_SEED_POSITIVES
            recall_curve.append(n_discovered / max(1, n_total_hidden))

        if _round == N_ROUNDS:
            break

        # Acquire — oracle reveals true label (positive or negative) for chosen genes
        if not has_both or strategy == "random":
            chosen = acquire_random(pool_idx, k=QUERY_SIZE, rng=rng)
        elif strategy == "uncertainty":
            chosen = acquire_uncertainty(pool_idx, model, X, k=QUERY_SIZE)
        else:  # qbc
            chosen = acquire_qbc(pool_idx, committee, X, k=QUERY_SIZE)

        labeled_idx = np.concatenate([labeled_idx, chosen])
        pool_idx = np.setdiff1d(pool_idx, chosen)

    return {"ap": ap_curve, "recall_top_k": recall_curve}


def run_strategy(gene_df: pd.DataFrame, feat_cols: list[str], strategy: str) -> dict:
    print(f"  [{strategy}] running {N_TRIALS} trials ...")
    all_ap, all_recall = [], []
    for t in range(N_TRIALS):
        rng = np.random.default_rng(RANDOM_STATE + t)
        trial = run_trial(gene_df, feat_cols, strategy, rng)
        all_ap.append(trial["ap"])
        all_recall.append(trial["recall_top_k"])

    ap = np.array(all_ap)
    recall = np.array(all_recall)
    return {
        "ap_mean": ap.mean(0),
        "ap_std": ap.std(0),
        "recall_mean": recall.mean(0),
        "recall_std": recall.std(0),
    }


# -- Plots ---------------------------------------------------------------------

def plot_curves(results: dict[str, dict]) -> None:
    rounds = np.arange(N_ROUNDS + 1)
    labeled_counts = (N_SEED_POSITIVES + N_SEED_NEGATIVES) + rounds * QUERY_SIZE
    colors = {"random": "gray", "uncertainty": "steelblue", "qbc": "darkorange"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Exp 3 — Active Learning vs Random  "
        f"({N_TRIALS} trials, query={QUERY_SIZE}/round)",
        fontsize=12, fontweight="bold",
    )

    for strategy, res in results.items():
        c = colors.get(strategy, "orange")
        axes[0].plot(labeled_counts, res["ap_mean"], label=strategy, color=c, lw=2)
        axes[0].fill_between(
            labeled_counts,
            res["ap_mean"] - res["ap_std"],
            res["ap_mean"] + res["ap_std"],
            alpha=0.2, color=c,
        )
        axes[1].plot(labeled_counts, res["recall_mean"], label=strategy, color=c, lw=2)
        axes[1].fill_between(
            labeled_counts,
            res["recall_mean"] - res["recall_std"],
            res["recall_mean"] + res["recall_std"],
            alpha=0.2, color=c,
        )

    axes[0].set_title("Average Precision on held-out test set\n(primary metric at 0.29% class imbalance)")
    axes[0].set_xlabel("Labeled genes (budget spent)")
    axes[0].set_ylabel("Average Precision")
    axes[0].legend()
    axes[0].set_ylim(0.0, 1.0)

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
    for strategy in ("random", "uncertainty", "qbc"):
        results[strategy] = run_strategy(gene_df, feat_cols, strategy)

    # Save summary
    summary_rows = []
    for strategy, res in results.items():
        summary_rows.append({
            "strategy": strategy,
            "final_ap_mean": res["ap_mean"][-1],
            "final_ap_std": res["ap_std"][-1],
            "final_recall_mean": res["recall_mean"][-1],
            "final_recall_std": res["recall_std"][-1],
            "auc_ap_curve": float(np.trapezoid(res["ap_mean"])),
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
