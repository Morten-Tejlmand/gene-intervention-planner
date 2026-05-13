"""
Experiment 6: Positive-Unlabeled (PU) Learning
================================================
Problem framing: We have 14 *confirmed* neural genes (positives) and ~28k
*unlabeled* genes — not confirmed negatives, just untested.  Standard
RandomForest treats every unlabeled gene as a true negative, which biases
the decision boundary against low-frequency positives.

PU Bagging (Mordelet & Vert, 2014):
  Train n_bags classifiers, each on:
    * ALL positive examples
    * A random subsample of unlabeled examples (treated as negatives per bag)
  Average predicted P(neural=1) across bags.
  Because each bag sees a different random negative set, the ensemble learns
  to distinguish positives from *typical* unlabeled genes rather than memorising
  specific confirmed-negative examples.

Strategies compared (both use uncertainty-sampling acquisition):
  standard_rf  — single RandomForest, class_weight="balanced"
  pu_bagging   — PU Bagging with RF base estimators

Metrics per round (10 trials):
  AUROC on fixed held-out test set
  Neural-gene recall in top-50 model predictions on the unlabeled pool

Run from project root:
    python experiments/exp06_pu_learning.py
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# -- Configuration -------------------------------------------------------------
DATA_PATH = Path("data/processed/enriched_v2_features.csv")
RESULTS_DIR = Path("artifacts/exp06_pu_learning")
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

# PU Bagging: number of bags and how many unlabeled to sample per bag.
# Sampling ~20x positives per bag keeps each bag balanced enough to learn
# while still seeing diverse negatives across bags.
PU_N_BAGS = 15
PU_NEG_RATIO = 20  # unlabeled sampled per bag = n_positives * PU_NEG_RATIO

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


# -- PU Bagging ----------------------------------------------------------------

class PUBaggingClassifier:
    """
    Positive-Unlabeled bagging (Mordelet & Vert 2014).

    Each bag trains an RF on all labeled positives plus a random subsample of
    unlabeled examples (y=0).  Final probabilities are averaged across bags,
    which smooths out the artefacts introduced by treating random unlabeled
    subsets as negatives.
    """

    def __init__(
        self,
        n_bags: int = PU_N_BAGS,
        neg_ratio: int = PU_NEG_RATIO,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_bags = n_bags
        self.neg_ratio = neg_ratio
        self.random_state = random_state
        self.estimators_: list[RandomForestClassifier] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PUBaggingClassifier":
        rng = np.random.default_rng(self.random_state)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_sample = min(len(pos_idx) * self.neg_ratio, len(neg_idx))

        self.estimators_ = []
        for i in range(self.n_bags):
            bag_neg = rng.choice(neg_idx, size=n_sample, replace=False)
            bag_idx = np.concatenate([pos_idx, bag_neg])
            clf = RandomForestClassifier(
                n_estimators=50,
                class_weight="balanced",
                random_state=int(self.random_state + i),
                n_jobs=-1,
            )
            clf.fit(X[bag_idx], y[bag_idx])
            self.estimators_.append(clf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        avg = np.mean(
            [e.predict_proba(X)[:, 1] for e in self.estimators_], axis=0
        )
        return np.column_stack([1.0 - avg, avg])


# -- Data ----------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA_PATH)
    neural_ids = set(NEURAL_GENE_IDS.keys()) | set(NEURAL_GENE_IDS.values())
    df["is_neural"] = df["common_name"].isin(neural_ids).astype(int)

    aa_cols = sorted(c for c in df.columns if c.startswith("percent_"))
    feat_cols = [c for c in FEATURE_COLS_BIOPHYS + aa_cols + FUNCTIONAL_FEATURES if c in df.columns]

    agg = {c: "mean" for c in feat_cols}
    agg["is_neural"] = "max"
    gene_df = df.groupby("common_name").agg(agg).reset_index()
    return gene_df, feat_cols


# -- Acquisition (shared) ------------------------------------------------------

def acquire_random(pool_idx: np.ndarray, *, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(pool_idx, size=min(k, len(pool_idx)), replace=False)


def acquire_uncertainty(pool_idx: np.ndarray, model, X: np.ndarray, *, k: int) -> np.ndarray:
    proba = model.predict_proba(X[pool_idx])
    if proba.shape[1] == 2:
        margin = np.abs(proba[:, 1] - proba[:, 0])
    else:
        s = np.sort(proba, axis=1)
        margin = s[:, -1] - s[:, -2]
    return pool_idx[np.argsort(margin)[:k]]


# -- Simulation ----------------------------------------------------------------

def _make_model(model_type: str):
    if model_type == "pu_bagging":
        return PUBaggingClassifier()
    return RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def run_trial(
    gene_df: pd.DataFrame,
    feat_cols: list[str],
    model_type: str,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    y = gene_df["is_neural"].values
    X = StandardScaler().fit_transform(gene_df[feat_cols].fillna(0.0).values)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) < N_SEED_POSITIVES + N_TEST_POSITIVES:
        raise ValueError(
            f"Not enough positives: need {N_SEED_POSITIVES + N_TEST_POSITIVES}, "
            f"got {len(pos_idx)}"
        )

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
    # Hidden positives are queryable — oracle reveals their label when selected
    pool_idx = np.concatenate([pool_neg, hidden_pos])
    n_total_hidden = len(hidden_pos)

    ap_curve: list[float] = []
    recall_curve: list[float] = []

    for _round in range(N_ROUNDS + 1):
        has_both = len(np.unique(y[labeled_idx])) >= 2

        model = _make_model(model_type)
        if has_both:
            model.fit(X[labeled_idx], y[labeled_idx])

        if not has_both:
            ap_curve.append(0.0)
            recall_curve.append(0.0)
        else:
            y_prob = model.predict_proba(X[test_idx])[:, -1]
            ap_curve.append(float(average_precision_score(y[test_idx], y_prob)))

            # Recall: hidden positives discovered so far via querying
            n_discovered = int(y[labeled_idx].sum()) - N_SEED_POSITIVES
            recall_curve.append(n_discovered / max(1, n_total_hidden))

        if _round == N_ROUNDS:
            break

        if not has_both:
            chosen = acquire_random(pool_idx, k=QUERY_SIZE, rng=rng)
        else:
            chosen = acquire_uncertainty(pool_idx, model, X, k=QUERY_SIZE)

        labeled_idx = np.concatenate([labeled_idx, chosen])
        pool_idx = np.setdiff1d(pool_idx, chosen)

    return {"ap": ap_curve, "recall_top_k": recall_curve}


def run_model_type(gene_df: pd.DataFrame, feat_cols: list[str], model_type: str) -> dict:
    print(f"  [{model_type}] {N_TRIALS} trials ...")
    all_ap, all_recall = [], []
    for t in range(N_TRIALS):
        rng = np.random.default_rng(RANDOM_STATE + t)
        trial = run_trial(gene_df, feat_cols, model_type, rng)
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
    colors = {"standard_rf": "steelblue", "pu_bagging": "seagreen"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Exp 6 — PU Bagging vs Standard RF  "
        f"({N_TRIALS} trials, {PU_N_BAGS} bags, neg_ratio={PU_NEG_RATIO}x)",
        fontsize=12, fontweight="bold",
    )

    for mtype, res in results.items():
        c = colors.get(mtype, "orange")
        axes[0].plot(labeled_counts, res["ap_mean"], label=mtype, color=c, lw=2)
        axes[0].fill_between(
            labeled_counts,
            res["ap_mean"] - res["ap_std"],
            res["ap_mean"] + res["ap_std"],
            alpha=0.2, color=c,
        )
        axes[1].plot(labeled_counts, res["recall_mean"], label=mtype, color=c, lw=2)
        axes[1].fill_between(
            labeled_counts,
            res["recall_mean"] - res["recall_std"],
            res["recall_mean"] + res["recall_std"],
            alpha=0.2, color=c,
        )

    axes[0].set_title("Average Precision on held-out test set\n(primary metric at 0.29% imbalance)")
    axes[0].set_xlabel("Labeled genes (budget spent)")
    axes[0].set_ylabel("Average Precision")
    axes[0].legend()
    axes[0].set_ylim(0.0, 1.0)

    axes[1].set_title(f"Neural-gene recall in top-{TOP_K} model ranking")
    axes[1].set_xlabel("Labeled genes (budget spent)")
    axes[1].set_ylabel(f"Recall@{TOP_K}")
    axes[1].legend()

    plt.tight_layout()
    out = RESULTS_DIR / "exp06_pu_vs_standard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot -> {out}")
    plt.close()


# -- Main ----------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Experiment 6: PU Bagging vs Standard RF")
    print("=" * 60)
    print(f"  PU bags={PU_N_BAGS}  neg_ratio={PU_NEG_RATIO}x  Rounds={N_ROUNDS}  Trials={N_TRIALS}")

    print(f"\nLoading {DATA_PATH} ...")
    gene_df, feat_cols = load_data()
    n_pos = int(gene_df["is_neural"].sum())
    print(f"  Genes: {len(gene_df):,}  |  Neural-relevant: {n_pos}")

    if n_pos < N_SEED_POSITIVES + N_TEST_POSITIVES:
        sys.exit(
            f"ERROR: Need {N_SEED_POSITIVES + N_TEST_POSITIVES} positives, found {n_pos}."
        )

    results: dict[str, dict] = {}
    for mtype in ("standard_rf", "pu_bagging"):
        results[mtype] = run_model_type(gene_df, feat_cols, mtype)

    summary_rows = []
    for mtype, res in results.items():
        summary_rows.append({
            "model": mtype,
            "final_ap_mean": res["ap_mean"][-1],
            "final_ap_std": res["ap_std"][-1],
            "final_recall_mean": res["recall_mean"][-1],
            "final_recall_std": res["recall_std"][-1],
            "auc_ap_curve": float(np.trapezoid(res["ap_mean"])),
        })
        pd.DataFrame({k: v for k, v in res.items() if isinstance(v, np.ndarray)}).to_csv(
            RESULTS_DIR / f"curves_{mtype}.csv", index=False
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    print("\nSaving plots ...")
    plot_curves(results)

    print("\n" + "-" * 60)
    print(summary.to_string(index=False))
    print("-" * 60)
    print(f"\nOutputs -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
