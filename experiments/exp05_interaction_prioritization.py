"""
Experiment 5: Gene Interaction Prioritization
==============================================
Goal: Move from single-gene selection to a pairwise combinatorial setting.
      Use the trained model to rank which gene combinations are most worth
      testing in the lab, reducing the combinatorial search space.

Focus: The synaptic adhesion cluster around nlg-1, nrx-1, and shn-1.
       We build candidate single and pairwise perturbation combinations
       from the top-N model-predicted neural genes and score them.

Scoring approach per pair (A, B):
  - Model relevance:   mean predicted neural probability
  - Biophysical complementarity:
      - hydrophobic fit      = 1 / (1 + |GRAVY_A - GRAVY_B|)
      - electrostatic drive  = |pI_A - pI_B|          (charge difference -> attractive)
      - joint stability      = 1 / (1 + instability_A x instability_B / 1000)
      - conservation match   = mean human_alignment_score
  - Composite priority score (tunable weights in SCORE_WEIGHTS)

Metrics vs random baseline:
  - Top-K hit rate: proportion of high-priority pairs containing a known
    synaptic gene (nlg-1, nrx-1, shn-1, unc-13, etc.)
  - Reduction in search space: how many pairs can be pruned

Run from project root:
    python experiments/exp05_interaction_prioritization.py
"""
from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -- Configuration -------------------------------------------------------------
DATA_PATH = Path("data/processed/enriched_genomic_features.csv")
RESULTS_DIR = Path("artifacts/exp05_interaction_prioritization")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TOP_N_GENES = 60          # candidate pool: top-N neural predictions + all seed genes
N_PAIRS_DISPLAY = 20      # top pairs to show in table

# Composite score weights: [model_relevance, hydro_fit, charge_drive, stability, conservation]
SCORE_WEIGHTS = (0.35, 0.15, 0.20, 0.15, 0.15)

# Known synaptic genes used as "ground truth" for evaluation
SYNAPTIC_ANCHOR_IDS = {
    "WBGene00003734": "nlg-1",
    "WBGene00003816": "nrx-1",
    "WBGene00019756": "shn-1",
    "WBGene00006745": "unc-13",
    "WBGene00004354": "ric-4",
    "WBGene00004944": "snb-1",
}

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


# -- Data & Model --------------------------------------------------------------

def load_and_train() -> tuple[pd.DataFrame, RandomForestClassifier, list[str]]:
    """Load data, aggregate to gene level, train RF on full dataset."""
    df = pd.read_csv(DATA_PATH)
    df["is_neural"] = df["common_name"].isin(NEURAL_GENE_IDS).astype(int)

    aa_cols = sorted(c for c in df.columns if c.startswith("percent_"))
    feat_cols = [c for c in FEATURE_COLS_BIOPHYS + aa_cols if c in df.columns]

    agg = {c: "mean" for c in feat_cols}
    agg["is_neural"] = "max"
    gene_df = df.groupby("common_name").agg(agg).reset_index()
    gene_df["gene_name"] = gene_df["common_name"].map(NEURAL_GENE_IDS).fillna(gene_df["common_name"])

    X = gene_df[feat_cols].fillna(0.0)
    y = gene_df["is_neural"]

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_sc, y)

    gene_df["neural_prob"] = model.predict_proba(X_sc)[:, -1]
    gene_df["_scaler"] = None
    gene_df.attrs["scaler"] = scaler

    return gene_df, model, feat_cols


# -- Candidate Pool ------------------------------------------------------------

def build_candidate_pool(gene_df: pd.DataFrame) -> pd.DataFrame:
    """Top-N by predicted neural probability + all seed genes."""
    seed_mask = gene_df["common_name"].isin(NEURAL_GENE_IDS)
    top_n = gene_df.nlargest(TOP_N_GENES, "neural_prob")
    pool = pd.concat([top_n, gene_df[seed_mask]]).drop_duplicates("common_name")
    print(f"  Candidate pool: {len(pool)} genes  "
          f"(top-{TOP_N_GENES} predicted + all {seed_mask.sum()} seed genes)")
    return pool.reset_index(drop=True)


# -- Pair Scoring --------------------------------------------------------------

def score_pairs(pool: pd.DataFrame) -> pd.DataFrame:
    """Build all pairwise combinations and compute composite priority scores."""
    w_rel, w_hyd, w_chg, w_stab, w_cons = SCORE_WEIGHTS
    records = []

    pool_vals = pool.set_index("common_name")
    genes = pool["common_name"].tolist()

    for g_a, g_b in combinations(genes, 2):
        a = pool_vals.loc[g_a]
        b = pool_vals.loc[g_b]

        # --- Biophysical components ---
        hydro_fit = 1.0 / (1.0 + abs(a["gravy"] - b["gravy"]))

        charge_drive = abs(a["isoelectric_point"] - b["isoelectric_point"])
        # Normalise charge drive to [0,1] relative to max possible (14.0 pH units)
        charge_drive_norm = min(charge_drive / 14.0, 1.0)

        joint_stability = 1.0 / (
            1.0 + (a["instability_index"] * b["instability_index"]) / 1000.0
        )

        conservation = float(
            np.mean([
                a.get("human_alignment_score", 0.0),
                b.get("human_alignment_score", 0.0),
            ])
        )
        # Normalise conservation score (alignment scores ~0–100)
        conservation_norm = min(conservation / 100.0, 1.0)

        # Mean predicted neural probability
        model_rel = float(np.mean([a["neural_prob"], b["neural_prob"]]))

        composite = (
            w_rel  * model_rel
            + w_hyd  * hydro_fit
            + w_chg  * charge_drive_norm
            + w_stab * joint_stability
            + w_cons * conservation_norm
        )

        # Is either gene a known synaptic anchor?
        is_anchor_pair = int(
            g_a in SYNAPTIC_ANCHOR_IDS or g_b in SYNAPTIC_ANCHOR_IDS
        )

        records.append({
            "gene_a": g_a,
            "name_a": a["gene_name"],
            "gene_b": g_b,
            "name_b": b["gene_name"],
            "model_relevance": model_rel,
            "hydro_fit": hydro_fit,
            "charge_drive": charge_drive_norm,
            "joint_stability": joint_stability,
            "conservation": conservation_norm,
            "composite_score": composite,
            "is_anchor_pair": is_anchor_pair,
        })

    pairs_df = pd.DataFrame(records).sort_values("composite_score", ascending=False).reset_index(drop=True)
    pairs_df["rank"] = pairs_df.index + 1
    return pairs_df


# -- Evaluation vs Random -------------------------------------------------------

def evaluate_vs_random(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Compare top-K hit rate (anchor pairs) vs random baseline."""
    n_total = len(pairs_df)
    n_anchor = int(pairs_df["is_anchor_pair"].sum())
    random_rate = n_anchor / n_total if n_total > 0 else 0.0

    ks = [10, 20, 50, 100, 200]
    rows = []
    for k in ks:
        if k > n_total:
            continue
        top_k = pairs_df.head(k)
        model_hits = int(top_k["is_anchor_pair"].sum())
        model_rate = model_hits / k
        random_hits = round(random_rate * k)
        rows.append({
            "K": k,
            "model_hits": model_hits,
            "model_rate": model_rate,
            "random_hits": random_hits,
            "random_rate": random_rate,
            "enrichment_fold": model_rate / random_rate if random_rate > 0 else float("inf"),
        })

    return pd.DataFrame(rows)


# -- Plots ---------------------------------------------------------------------

def plot_results(pairs_df: pd.DataFrame, eval_df: pd.DataFrame, pool: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Exp 5 — Gene Interaction Prioritization", fontsize=12, fontweight="bold")

    # 1. Score distribution: model vs random
    axes[0].hist(pairs_df["composite_score"], bins=40, alpha=0.7, color="steelblue", label="All pairs")
    axes[0].hist(pairs_df[pairs_df["is_anchor_pair"] == 1]["composite_score"],
                 bins=20, alpha=0.7, color="darkorange", label="Anchor pairs")
    axes[0].set_title("Composite Score Distribution")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # 2. Top-K hit rate vs random
    if not eval_df.empty:
        x = eval_df["K"]
        axes[1].plot(x, eval_df["model_rate"], "o-", color="steelblue", label="Model (ranked)", lw=2)
        axes[1].plot(x, eval_df["random_rate"], "s--", color="gray", label="Random", lw=2)
        axes[1].set_title("Top-K Anchor-Pair Hit Rate")
        axes[1].set_xlabel("K (top pairs examined)")
        axes[1].set_ylabel("Proportion containing anchor gene")
        axes[1].legend()

    # 3. Candidate pool: neural_prob coloured by anchor status
    colors_pool = ["darkorange" if g in SYNAPTIC_ANCHOR_IDS else "steelblue"
                   for g in pool["common_name"]]
    axes[2].barh(pool["gene_name"].head(30), pool["neural_prob"].head(30),
                 color=colors_pool[:30])
    axes[2].invert_yaxis()
    axes[2].set_title("Top-30 Candidate Genes\n(orange = known synaptic anchor)")
    axes[2].set_xlabel("Predicted neural probability")

    plt.tight_layout()
    out = RESULTS_DIR / "exp05_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot -> {out}")
    plt.close()


# -- Main ----------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Experiment 5: Gene Interaction Prioritization")
    print("=" * 60)
    print(f"  Score weights (rel, hydro, charge, stab, cons) = {SCORE_WEIGHTS}")

    print(f"\nLoading {DATA_PATH} and training model ...")
    gene_df, model, feat_cols = load_and_train()
    del gene_df["_scaler"]
    n_pos = int(gene_df["is_neural"].sum())
    print(f"  Genes: {len(gene_df):,}  |  Neural-relevant (seeds): {n_pos}")

    print("\nBuilding candidate pool ...")
    pool = build_candidate_pool(gene_df)

    print(f"Building pairwise combinations from {len(pool)} candidates ...")
    pairs_df = score_pairs(pool)
    n_pairs = len(pairs_df)
    print(f"  Total pairs: {n_pairs:,}  |  Anchor pairs: {pairs_df['is_anchor_pair'].sum()}")

    print("\nEvaluating vs random baseline ...")
    eval_df = evaluate_vs_random(pairs_df)

    # Save outputs
    pairs_df.to_csv(RESULTS_DIR / "ranked_pairs.csv", index=False)
    eval_df.to_csv(RESULTS_DIR / "topk_evaluation.csv", index=False)
    pool.to_csv(RESULTS_DIR / "candidate_pool.csv", index=False)

    print("\nSaving plots ...")
    plot_results(pairs_df, eval_df, pool)

    print("\n" + "-" * 60)
    print(f"Top {N_PAIRS_DISPLAY} prioritized gene pairs:")
    display_cols = ["rank", "name_a", "name_b", "model_relevance", "composite_score", "is_anchor_pair"]
    print(pairs_df[display_cols].head(N_PAIRS_DISPLAY).to_string(index=False))

    print("\nTop-K evaluation vs random:")
    print(eval_df.to_string(index=False))
    print("-" * 60)
    print(f"\nOutputs -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
