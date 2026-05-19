# Exp04 - Hybrid Acquisition Strategies
# Same AL setup as exp03 but focused on comparing acquisition strategies more carefully
# Key question: does UCB (exploration + exploitation) beat pure uncertainty (margin)?
#
# Strategies:
#   random  = no model, uniform sampling
#   margin  = pure uncertainty: pick genes where P(positive) ~ 0.5
#   ucb     = P(positive) + beta * uncertainty (balance explore vs exploit)
#
# TODO: try different beta values for UCB

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
from data_loader import load_gene_level_data
from models import GBTClassifier

try:
    from pub_style import apply_pub_style
    apply_pub_style()
except ImportError:
    pass

os.makedirs("artifacts/behavioural_al/exp04_hybrid", exist_ok=True)
RESULTS_DIR = Path("artifacts/behavioural_al/exp04_hybrid")

TARGET_COL = "neural_behaviour_count"
THRESHOLD = 1
N_SEED = 20
N_ROUNDS = 25
QUERY_SIZE = 10
N_TRIALS = 5
N_ESTIMATORS = 20
RANDOM_STATE = 42
UCB_BETA = 1.0

ALL_STRATEGIES = ["random", "margin", "ucb"]

STRATEGY_COLORS = {
    "random": "#888888",
    "margin": "#FF9800",
    "ucb": "#2196F3",
}

STRATEGY_LABELS = {
    "random": "Random (baseline)",
    "margin": "Margin (uncertainty)",
    "ucb": "UCB (balanced)",
}


def run_trial(X, y, strategy, rng, confirmed_mask):
    """Run a single AL trial. Returns AP and recall curve."""
    pos_idx = np.where(y == 1)[0]
    confirmed_neg_idx = np.where((y == 0) & confirmed_mask)[0]
    unknown_idx = np.where(~confirmed_mask)[0]

    rng.shuffle(pos_idx := pos_idx.copy())
    n_test_pos = max(5, len(pos_idx) // 5)
    test_pos = pos_idx[:n_test_pos]
    hidden_pos = pos_idx[n_test_pos:]

    rng.shuffle(confirmed_neg_idx := confirmed_neg_idx.copy())
    n_test_neg = n_test_pos * 4
    test_neg = confirmed_neg_idx[:n_test_neg]
    pool_neg = confirmed_neg_idx[n_test_neg:]

    test_idx = np.concatenate([test_pos, test_neg])

    available_for_seed = np.concatenate([hidden_pos, pool_neg])
    seed_size = min(N_SEED, len(available_for_seed))
    seed_idx = rng.choice(available_for_seed, size=seed_size, replace=False)

    pool_idx = np.setdiff1d(np.concatenate([available_for_seed, unknown_idx]), seed_idx)
    labeled_idx = seed_idx.copy()

    n_hidden_pos = len(hidden_pos)
    acq_fn = STRATEGIES[strategy]

    ap_curve = []
    recall_curve = []

    for round_num in range(N_ROUNDS + 1):
        has_both_classes = len(np.unique(y[labeled_idx])) >= 2

        if has_both_classes:
            model = GBTClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
            model.fit(X[labeled_idx], y[labeled_idx])
            proba_test, _ = model.predict(X[test_idx])
            ap = float(average_precision_score(y[test_idx], proba_test))
        else:
            ap = 0.0

        discovered = int(np.isin(hidden_pos, labeled_idx).sum())
        ap_curve.append(ap)
        recall_curve.append(discovered / max(1, n_hidden_pos))

        if round_num == N_ROUNDS:
            break

        if not has_both_classes or len(pool_idx) == 0:
            continue

        proba_pool, std_pool = model.predict(X[pool_idx])

        if strategy == "ucb":
            scores = acq_fn(proba_pool, std_pool, beta=UCB_BETA)
        else:
            scores = acq_fn(proba_pool, std_pool)

        k = min(QUERY_SIZE, len(pool_idx))
        chosen = pool_idx[np.argsort(scores)[::-1][:k]]

        chosen_confirmed = chosen[confirmed_mask[chosen]]
        labeled_idx = np.concatenate([labeled_idx, chosen_confirmed])
        pool_idx = np.setdiff1d(pool_idx, chosen)

    return {"ap": ap_curve, "recall": recall_curve}


def run_strategy(X, y, strategy, confirmed_mask):
    print(f"  [{strategy:12s}] running {N_TRIALS} trials ...")
    all_ap = []
    all_recall = []

    for t in range(N_TRIALS):
        rng = np.random.default_rng(RANDOM_STATE + t)
        trial = run_trial(X, y, strategy, rng, confirmed_mask)
        all_ap.append(trial["ap"])
        all_recall.append(trial["recall"])

    ap = np.array(all_ap)
    rec = np.array(all_recall)

    return {
        "ap_mean": ap.mean(0),
        "ap_std": ap.std(0),
        "recall_mean": rec.mean(0),
        "recall_std": rec.std(0),
        "aulc": float(np.trapezoid(ap.mean(0))),
        "final_ap_trials": ap[:, -1],
        "final_rec_trials": rec[:, -1],
    }


def plot_results(results, n_pos, n_confirmed, n_unknown):
    rounds = np.arange(len(next(iter(results.values()))["ap_mean"]))
    budget = N_SEED + rounds * QUERY_SIZE

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Exp04 - Hybrid Acquisition (threshold>={THRESHOLD})  |  "
        f"positives: {n_pos:,}  confirmed_neg: {n_confirmed - n_pos:,}  unknown: {n_unknown:,}",
        fontweight="bold",
    )

    # AP curves
    for s, res in results.items():
        c = STRATEGY_COLORS[s]
        label = STRATEGY_LABELS[s]
        axes[0].plot(budget, res["ap_mean"], label=label, color=c, linewidth=1.8)
        axes[0].fill_between(
            budget,
            np.clip(res["ap_mean"] - res["ap_std"], 0, 1),
            np.clip(res["ap_mean"] + res["ap_std"], 0, 1),
            alpha=0.15, color=c,
        )
    axes[0].set(title="Average Precision (held-out test)",
                xlabel="Labeled genes (budget)", ylabel="AP", ylim=(0, 1))
    axes[0].legend(fontsize=8)

    # recall curves
    for s, res in results.items():
        c = STRATEGY_COLORS[s]
        label = STRATEGY_LABELS[s]
        axes[1].plot(budget, res["recall_mean"], label=label, color=c, linewidth=1.8)
        axes[1].fill_between(
            budget,
            np.clip(res["recall_mean"] - res["recall_std"], 0, 1),
            res["recall_mean"] + res["recall_std"],
            alpha=0.15, color=c,
        )
    axes[1].set(title="Recall (hidden positives discovered)",
                xlabel="Labeled genes (budget)", ylabel="Recall")
    axes[1].legend(fontsize=8)

    # AULC vs final AP side-by-side bars
    strategies = list(results.keys())
    aulc_vals = [results[s]["aulc"] for s in strategies]
    final_ap = [results[s]["ap_mean"][-1] for s in strategies]
    final_err = [results[s]["ap_std"][-1] for s in strategies]
    x = np.arange(len(strategies))
    width = 0.38

    bars_aulc = axes[2].bar(x - width / 2, aulc_vals, width=width,
                            color=[STRATEGY_COLORS[s] for s in strategies],
                            alpha=0.85, label="AULC")
    axes[2].bar(x + width / 2, final_ap, width=width, yerr=final_err,
                color=[STRATEGY_COLORS[s] for s in strategies],
                alpha=0.45, capsize=4, label="Final AP")

    for bar, v in zip(bars_aulc, aulc_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     v + 0.01 * max(aulc_vals),
                     f"{v:.2f}", ha="center", fontsize=8)

    axes[2].set(title="AULC vs Final AP\n(dark=AULC, light=final AP)",
                xticks=x, xticklabels=strategies)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "hybrid_strategies.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Plot -> {out}")
    plt.close()


def main():
    print("=" * 60)
    print("Exp04 - Hybrid Acquisition Strategies")
    print(f"  Target   : {TARGET_COL} >= {THRESHOLD}")
    print(f"  Rounds   : {N_ROUNDS}  |  Query/round: {QUERY_SIZE}  |  Seed: {N_SEED}")
    print(f"  Trials   : {N_TRIALS}  |  Strategies: {', '.join(ALL_STRATEGIES)}")
    print("=" * 60)

    print("\nLoading data ...")
    gene_df, feat_cols = load_gene_level_data(
        target_col=TARGET_COL, include_esm2=False, include_go=False,
    )

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
    for strategy in ALL_STRATEGIES:
        results[strategy] = run_strategy(X, y, strategy, confirmed_mask)

    # pairwise t-tests
    print("\nPairwise t-tests on final AP (two-sided):")
    for i, s1 in enumerate(ALL_STRATEGIES):
        for s2 in ALL_STRATEGIES[i + 1:]:
            t, p = stats.ttest_ind(results[s1]["final_ap_trials"], results[s2]["final_ap_trials"])
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            print(f"  {s1:12s} vs {s2:12s}: t={t:+.3f}  p={p:.4f}  {sig}")

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
    summary.to_csv(RESULTS_DIR / "strategy_summary.csv", index=False)

    for s, res in results.items():
        pd.DataFrame({
            "ap_mean": res["ap_mean"],
            "ap_std": res["ap_std"],
            "recall_mean": res["recall_mean"],
            "recall_std": res["recall_std"],
        }).to_csv(RESULTS_DIR / f"curve_{s}.csv", index=False)

    print("\nSaving plots ...")
    plot_results(results, n_pos, n_confirmed, n_unknown)

    print("\n" + "-" * 60)
    print(summary.to_string(index=False))
    print("-" * 60)
    print(f"\nOutputs -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
