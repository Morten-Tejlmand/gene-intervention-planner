# Exp03 - Active Learning vs Random Sampling
# Main question: does model-guided gene selection find neural-behavioural genes
# faster than random sampling with the same budget?
#
# Uses GBT (best model from exp01/02) with 3 strategies:
#   random  = baseline, no model guidance
#   margin  = query genes where model is most uncertain (P ~ 0.5)
#   ucb     = P(positive) + beta*uncertainty (exploit + explore)
#
# Unknown genes (no WormBase annotation) can be queried but give no label - "wasted" queries
# TODO: could try weighting the pool to avoid unknowns

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
from scipy import stats
from sklearn.metrics import average_precision_score

from acquisition import STRATEGIES
from data_loader import BIOPHYS_FEATURES, ORTHOLOG_FEATURES, load_gene_level_data
from models import GBTClassifier

try:
    from pub_style import apply_pub_style
    apply_pub_style()
except ImportError:
    pass

os.makedirs("artifacts/behavioural_al/exp03_al_vs_random", exist_ok=True)
RESULTS_DIR = Path("artifacts/behavioural_al/exp03_al_vs_random")

TARGET_COL = "neural_behaviour_count"
THRESHOLD = 1
N_SEED = 20       # starting labeled set size
N_ROUNDS = 50     # number of AL rounds
QUERY_SIZE = 10   # genes queried per round
N_TRIALS = 10     # independent trials (different random seeds)
N_ESTIMATORS = 20 # trees per model - lower than default for speed
RANDOM_STATE = 42
UCB_BETA = 1.0    # exploration weight for UCB strategy

EXP02_CSV = Path("artifacts/behavioural_al/exp02_feature_comparison/feature_comparison.csv")

STRATEGY_COLORS = {
    "random": "#888888",
    "margin": "#FF9800",
    "ucb": "#2196F3",
}


def get_best_feature_cols(gene_df, all_feat_cols):
    """Read exp02 results and return feature columns for the best AP feature set."""
    aa_cols = sorted([c for c in gene_df.columns if c.startswith("percent_")])
    biophys_cols = [c for c in BIOPHYS_FEATURES if c in gene_df.columns]
    if "human_alignment_score" in gene_df.columns:
        biophys_cols = biophys_cols + ["human_alignment_score"]
    ortholog_cols = [c for c in ORTHOLOG_FEATURES if c in gene_df.columns]
    esm2_cols = [c for c in all_feat_cols if c.startswith("esm2_")]
    go_cols = [c for c in all_feat_cols if c.startswith("go_svd_")]

    feat_map = {
        "A — Biophysical": biophys_cols,
        "B — +AA composition": biophys_cols + aa_cols,
        "C — +Evolutionary": biophys_cols + aa_cols + ortholog_cols,
        "D — +GO/ESM-2": biophys_cols + aa_cols + ortholog_cols + esm2_cols + go_cols,
    }

    # try to read exp02 results and use the best feature set
    if EXP02_CSV.exists():
        exp02 = pd.read_csv(EXP02_CSV)
        best_set = exp02.loc[exp02["ap"].idxmax(), "feature_set"]
        if best_set in feat_map:
            return feat_map[best_set], best_set

    # fallback to set C if exp02 hasn't run yet
    default = "C — +Evolutionary"
    return feat_map[default], default


def run_trial(X, y, strategy, rng, confirmed_mask):
    """Run a single AL trial. Returns AP curve, recall curve, and wasted queries per round."""
    pos_idx = np.where(y == 1)[0]
    confirmed_neg_idx = np.where((y == 0) & confirmed_mask)[0]
    unknown_idx = np.where(~confirmed_mask)[0]

    # hold out a test set (never queried, only used for evaluation)
    rng.shuffle(pos_idx := pos_idx.copy())
    n_test_pos = max(5, len(pos_idx) // 5)
    test_pos = pos_idx[:n_test_pos]
    hidden_pos = pos_idx[n_test_pos:]  # the positives the AL agent can discover

    rng.shuffle(confirmed_neg_idx := confirmed_neg_idx.copy())
    n_test_neg = n_test_pos * 4  # 4:1 neg:pos ratio in test set
    test_neg = confirmed_neg_idx[:n_test_neg]
    pool_neg = confirmed_neg_idx[n_test_neg:]

    test_idx = np.concatenate([test_pos, test_neg])

    # seed: small labeled set to start from (confirmed genes only)
    available_for_seed = np.concatenate([hidden_pos, pool_neg])
    seed_size = min(N_SEED, len(available_for_seed))
    seed_idx = rng.choice(available_for_seed, size=seed_size, replace=False)

    pool_idx = np.setdiff1d(np.concatenate([available_for_seed, unknown_idx]), seed_idx)
    labeled_idx = seed_idx.copy()

    n_hidden_pos = len(hidden_pos)
    acq_fn = STRATEGIES[strategy]

    ap_curve = []
    recall_curve = []
    wasted_curve = []

    for round_num in range(N_ROUNDS + 1):
        # need at least one positive and one negative to train
        has_both_classes = len(np.unique(y[labeled_idx])) >= 2

        if has_both_classes:
            model = GBTClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
            model.fit(X[labeled_idx], y[labeled_idx])
            proba_test, _ = model.predict(X[test_idx])
            ap = float(average_precision_score(y[test_idx], proba_test))
        else:
            ap = 0.0

        # how many of the hidden positives have we found so far?
        discovered = int(np.isin(hidden_pos, labeled_idx).sum())
        ap_curve.append(ap)
        recall_curve.append(discovered / max(1, n_hidden_pos))

        if round_num == N_ROUNDS:
            break

        if not has_both_classes or len(pool_idx) == 0:
            wasted_curve.append(0)
            continue

        # score the pool and pick top-k genes
        proba_pool, std_pool = model.predict(X[pool_idx])

        if strategy == "ucb":
            scores = acq_fn(proba_pool, std_pool, beta=UCB_BETA)
        else:
            scores = acq_fn(proba_pool, std_pool)

        k = min(QUERY_SIZE, len(pool_idx))
        chosen = pool_idx[np.argsort(scores)[::-1][:k]]

        # confirmed genes get added to labeled set; unknowns are just removed from pool
        chosen_confirmed = chosen[confirmed_mask[chosen]]
        chosen_unknown = chosen[~confirmed_mask[chosen]]

        labeled_idx = np.concatenate([labeled_idx, chosen_confirmed])
        pool_idx = np.setdiff1d(pool_idx, chosen)
        wasted_curve.append(len(chosen_unknown))

    return {"ap": ap_curve, "recall": recall_curve, "wasted": wasted_curve}


def run_strategy(X, y, strategy, confirmed_mask):
    print(f"  [{strategy}] running {N_TRIALS} trials ...")
    all_ap = []
    all_recall = []
    all_wasted = []

    for t in range(N_TRIALS):
        rng = np.random.default_rng(RANDOM_STATE + t)
        trial = run_trial(X, y, strategy, rng, confirmed_mask)
        all_ap.append(trial["ap"])
        all_recall.append(trial["recall"])
        all_wasted.append(trial["wasted"])

    ap = np.array(all_ap)
    rec = np.array(all_recall)
    max_w = max(len(w) for w in all_wasted)
    wasted = np.array([w + [0] * (max_w - len(w)) for w in all_wasted])

    return {
        "ap_mean": ap.mean(0),
        "ap_std": ap.std(0),
        "recall_mean": rec.mean(0),
        "recall_std": rec.std(0),
        "wasted_mean": wasted.mean(0),
        "aulc": float(np.trapezoid(ap.mean(0))),
        "final_ap_trials": ap[:, -1],
        "final_rec_trials": rec[:, -1],
    }


def plot_results(results, universe):
    rounds = np.arange(len(next(iter(results.values()))["ap_mean"]))
    budget = N_SEED + rounds * QUERY_SIZE

    n_pos = int((universe[TARGET_COL] >= THRESHOLD).sum())
    n_confirmed = int((universe["has_annotation"] > 0).sum())
    n_unknown = int((universe["has_annotation"] == 0).sum())

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle(
        f"Exp03 - AL vs Random (threshold>={THRESHOLD})  |  "
        f"positives: {n_pos:,}  confirmed_neg: {n_confirmed - n_pos:,}  unknown: {n_unknown:,}",
        fontweight="bold",
    )

    # AP learning curves
    for strategy, res in results.items():
        c = STRATEGY_COLORS.get(strategy, "#888888")
        axes[0].plot(budget, res["ap_mean"], label=strategy, color=c)
        axes[0].fill_between(
            budget,
            np.clip(res["ap_mean"] - res["ap_std"], 0, 1),
            np.clip(res["ap_mean"] + res["ap_std"], 0, 1),
            alpha=0.2, color=c,
        )
    axes[0].set(title="Average Precision (held-out test)",
                xlabel="Labeled genes (budget)", ylabel="AP", ylim=(0, 1))
    axes[0].legend()

    # recall curves - fraction of hidden positives discovered
    for strategy, res in results.items():
        c = STRATEGY_COLORS.get(strategy, "#888888")
        axes[1].plot(budget, res["recall_mean"], label=strategy, color=c)
        axes[1].fill_between(
            budget,
            np.clip(res["recall_mean"] - res["recall_std"], 0, 1),
            res["recall_mean"] + res["recall_std"],
            alpha=0.2, color=c,
        )
    axes[1].set(title="Recall (hidden positives discovered)",
                xlabel="Labeled genes (budget)", ylabel="Recall")
    axes[1].legend()

    # AULC bar chart
    strategies = list(results.keys())
    aulc_vals = [results[s]["aulc"] for s in strategies]
    bars = axes[2].bar(strategies, aulc_vals,
                       color=[STRATEGY_COLORS.get(s, "#888888") for s in strategies],
                       alpha=0.85)
    for bar, v in zip(bars, aulc_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     v + 0.02 * max(aulc_vals),
                     f"{v:.2f}", ha="center", fontsize=9)
    axes[2].set(title="AULC (area under AP curve)", ylabel="AULC")

    # wasted queries per round
    rounds_q = np.arange(1, N_ROUNDS + 1)
    random_expected = QUERY_SIZE * (n_unknown / len(universe))
    for strategy, res in results.items():
        c = STRATEGY_COLORS.get(strategy, "#888888")
        w = res["wasted_mean"]
        axes[3].plot(rounds_q[:len(w)], w, label=strategy, color=c)
    axes[3].axhline(random_expected, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
                    label=f"random expected ({random_expected:.2f})")
    axes[3].set(title="Wasted queries/round\n(unknown genes)",
                xlabel="Round", ylabel="Wasted queries")
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "al_vs_random.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Plot -> {out}")
    plt.close()


def main():
    print("=" * 60)
    print("Exp03 - Active Learning vs Random Sampling")
    print(f"  Target  : {TARGET_COL} >= {THRESHOLD}")
    print(f"  Rounds  : {N_ROUNDS}  |  Query/round: {QUERY_SIZE}  |  Seed: {N_SEED}")
    print(f"  Trials  : {N_TRIALS}")
    print("=" * 60)

    print("\nLoading data ...")
    gene_df, all_feat_cols = load_gene_level_data(
        target_col=TARGET_COL, include_esm2=True, include_go=True,
    )
    feat_cols, best_set = get_best_feature_cols(gene_df, all_feat_cols)
    print(f"  Feature set       : {best_set}  ({len(feat_cols)} features)")

    universe = gene_df.copy().reset_index(drop=True)
    universe["label"] = (universe[TARGET_COL] >= THRESHOLD).astype(int)

    confirmed_mask = universe["has_annotation"].values > 0
    n_pos = int(universe["label"].sum())
    n_confirmed = int(confirmed_mask.sum())
    n_unknown = int((~confirmed_mask).sum())

    print(f"  Universe          : {len(universe):,} genes")
    print(f"  Features          : {len(feat_cols)}")
    print(f"  Positives         : {n_pos:,} ({100*n_pos/len(universe):.1f}%)")
    print(f"  Confirmed negative: {n_confirmed - n_pos:,}")
    print(f"  Unknown           : {n_unknown:,}")

    X = universe[feat_cols].values.astype(float)
    y = universe["label"].values.astype(int)

    results = {}
    for strategy in ("random", "margin", "ucb"):
        results[strategy] = run_strategy(X, y, strategy, confirmed_mask)

    # statistical tests between strategies
    strategies = list(results.keys())
    print("\nPairwise t-tests on final AP (two-sided):")
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i + 1:]:
            t, p = stats.ttest_ind(results[s1]["final_ap_trials"], results[s2]["final_ap_trials"])
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            print(f"  {s1} vs {s2}: t={t:.3f}  p={p:.4f}  {sig}")

    # save summary
    summary = pd.DataFrame([
        {
            "strategy": s,
            "final_ap_mean": res["ap_mean"][-1],
            "final_ap_std": res["ap_std"][-1],
            "final_rec_mean": res["recall_mean"][-1],
            "final_rec_std": res["recall_std"][-1],
            "aulc": res["aulc"],
        }
        for s, res in results.items()
    ])
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    for s, res in results.items():
        pd.DataFrame({
            "ap_mean": res["ap_mean"],
            "ap_std": res["ap_std"],
            "recall_mean": res["recall_mean"],
            "recall_std": res["recall_std"],
        }).to_csv(RESULTS_DIR / f"curve_{s}.csv", index=False)

    print("\nSaving plots ...")
    plot_results(results, universe)

    print("\n" + "-" * 60)
    print(summary.to_string(index=False))
    print("-" * 60)
    print(f"\nOutputs -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
