"""
Behavioural Active Learning — Binary Classification
=====================================================
Core question: Does model-guided gene selection find neural-behavioural genes
faster than random sampling under a fixed experiment budget?

Problem framing
---------------
Target  : binary — neural_behaviour_count >= THRESHOLD → positive (1), else 0.
          Threshold >= 1: any documented neural behavioural effect.
          Threshold >= 2: at least two independent annotations (more reliable).
Model   : EnsembleClassifier (RF + GBT + ExtraTrees, class_weight=balanced).
          Returns P(positive) mean + std uncertainty across members.
Acquisition:
    random  — uniform baseline
    margin  — uncertainty sampling: select genes where P(positive) ~ 0.5
    ucb     — P(positive) + beta*std: exploit likely positives + explore uncertain

Label types
-----------
Confirmed positive  : neural_behaviour_count >= THRESHOLD
Confirmed negative  : has_annotation > 0  AND  neural_behaviour_count < THRESHOLD
                      (gene was studied for something — y=0 is reliable)
Unknown             : has_annotation == 0
                      (never in WormBase — y=0 means untested, not negative)
                      Queried genes from this group are removed from the pool
                      but NOT added to the labeled set (wasted query).

Metrics
-------
Average Precision   : on a fixed held-out test set each round
Recall              : fraction of hidden positives discovered so far
Wasted queries      : unknown genes queried per round

Run from project root:
    python experiments/behavioural_al/run_experiment.py
"""
from __future__ import annotations

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
from scipy import stats
from sklearn.metrics import average_precision_score

from acquisition import STRATEGIES
from data_loader import load_gene_level_data
from models import EnsembleClassifier

try:
    from pub_style import apply_pub_style
    apply_pub_style()
except ImportError:
    pass

# -- Configuration ----------------------------------------------------------
RESULTS_DIR = Path("artifacts/behavioural_al")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL   = "neural_behaviour_count"
THRESHOLD    = 1      # count >= THRESHOLD → positive. Try 1 or 2.
N_SEED       = 20     # labeled genes at start (drawn from confirmed only)
N_ROUNDS     = 25
QUERY_SIZE   = 10
N_TRIALS     = 5
N_ESTIMATORS = 20
RANDOM_STATE = 42
UCB_BETA     = 1.0    # exploration weight for UCB

USE_ESM2 = False
USE_GO   = False

STRATEGY_COLORS = {
    "random": "#888888",
    "margin": "#FF9800",
    "ucb":    "#2196F3",
}


# -- Simulation -------------------------------------------------------------

def run_trial(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
    confirmed_mask: np.ndarray,
) -> dict:
    """
    Single AL trial.

    Test set is held out from the start and never queried — used only for
    AP evaluation each round. Pool contains confirmed + unknown genes.
    Unknown genes that get queried are removed from pool but not labeled.
    """
    pos_idx           = np.where(y == 1)[0]
    confirmed_neg_idx = np.where((y == 0) & confirmed_mask)[0]
    unknown_idx       = np.where(~confirmed_mask)[0]

    # Hold out test set (never queried)
    rng.shuffle(pos_idx := pos_idx.copy())
    n_test_pos = max(5, len(pos_idx) // 5)
    test_pos   = pos_idx[:n_test_pos]
    hidden_pos = pos_idx[n_test_pos:]       # positives the AL agent can discover

    rng.shuffle(confirmed_neg_idx := confirmed_neg_idx.copy())
    n_test_neg = n_test_pos * 4             # 4:1 neg:pos ratio in test set
    test_neg   = confirmed_neg_idx[:n_test_neg]
    pool_neg   = confirmed_neg_idx[n_test_neg:]

    test_idx = np.concatenate([test_pos, test_neg])

    # Seed: small random set from confirmed genes (not from test or unknown)
    available_for_seed = np.concatenate([hidden_pos, pool_neg])
    seed_size  = min(N_SEED, len(available_for_seed))
    seed_idx   = rng.choice(available_for_seed, size=seed_size, replace=False)

    pool_idx    = np.setdiff1d(np.concatenate([available_for_seed, unknown_idx]), seed_idx)
    labeled_idx = seed_idx.copy()

    n_hidden_pos = len(hidden_pos)
    acq_fn       = STRATEGIES[strategy]

    ap_curve:     list[float] = []
    recall_curve: list[float] = []
    wasted_curve: list[int]   = []

    for _round in range(N_ROUNDS + 1):
        has_both = len(np.unique(y[labeled_idx])) >= 2

        if has_both:
            model = EnsembleClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
            model.fit(X[labeled_idx], y[labeled_idx])
            proba_test, _ = model.predict(X[test_idx])
            ap = float(average_precision_score(y[test_idx], proba_test))
        else:
            ap = 0.0

        discovered = int(np.isin(hidden_pos, labeled_idx).sum())
        ap_curve.append(ap)
        recall_curve.append(discovered / max(1, n_hidden_pos))

        if _round == N_ROUNDS:
            break

        if not has_both or len(pool_idx) == 0:
            wasted_curve.append(0)
            continue

        proba_pool, std_pool = model.predict(X[pool_idx])

        if strategy == "ucb":
            scores = acq_fn(proba_pool, std_pool, beta=UCB_BETA)
        else:
            scores = acq_fn(proba_pool, std_pool)

        k      = min(QUERY_SIZE, len(pool_idx))
        chosen = pool_idx[np.argsort(scores)[::-1][:k]]

        chosen_confirmed = chosen[confirmed_mask[chosen]]
        chosen_unknown   = chosen[~confirmed_mask[chosen]]

        labeled_idx = np.concatenate([labeled_idx, chosen_confirmed])
        pool_idx    = np.setdiff1d(pool_idx, chosen)

        wasted_curve.append(len(chosen_unknown))

    return {"ap": ap_curve, "recall": recall_curve, "wasted": wasted_curve}


def run_strategy(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str,
    confirmed_mask: np.ndarray,
) -> dict:
    print(f"  [{strategy}] running {N_TRIALS} trials ...")
    all_ap, all_recall, all_wasted = [], [], []

    for t in range(N_TRIALS):
        rng   = np.random.default_rng(RANDOM_STATE + t)
        trial = run_trial(X, y, strategy, rng, confirmed_mask)
        all_ap.append(trial["ap"])
        all_recall.append(trial["recall"])
        all_wasted.append(trial["wasted"])

    ap     = np.array(all_ap)
    rec    = np.array(all_recall)
    max_w  = max(len(w) for w in all_wasted)
    wasted = np.array([w + [0] * (max_w - len(w)) for w in all_wasted])

    return {
        "ap_mean":           ap.mean(0),
        "ap_std":            ap.std(0),
        "recall_mean":       rec.mean(0),
        "recall_std":        rec.std(0),
        "wasted_mean":       wasted.mean(0),
        "aulc":              float(np.trapezoid(ap.mean(0))),
        "final_ap_trials":   ap[:, -1],
        "final_rec_trials":  rec[:, -1],
    }


# -- Plots ------------------------------------------------------------------

def plot_results(results: dict[str, dict], universe: pd.DataFrame) -> None:
    rounds = np.arange(len(next(iter(results.values()))["ap_mean"]))
    budget = N_SEED + rounds * QUERY_SIZE

    n_pos       = int((universe[TARGET_COL] >= THRESHOLD).sum())
    n_confirmed = int((universe["has_annotation"] > 0).sum())
    n_unknown   = int((universe["has_annotation"] == 0).sum())

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle(
        f"Behavioural AL (binary, threshold>={THRESHOLD})  |  "
        f"positives: {n_pos:,}  confirmed_neg: {n_confirmed - n_pos:,}  unknown: {n_unknown:,}",
        fontweight="bold",
    )

    # 1. Average Precision
    for strategy, res in results.items():
        c = STRATEGY_COLORS.get(strategy, "#888888")
        axes[0].plot(budget, res["ap_mean"], label=strategy, color=c)
        axes[0].fill_between(
            budget,
            np.clip(res["ap_mean"] - res["ap_std"], 0, 1),
            np.clip(res["ap_mean"] + res["ap_std"], 0, 1),
            alpha=0.2, color=c,
        )
    axes[0].set(
        title="Average Precision (held-out test)",
        xlabel="Labeled genes (budget)", ylabel="AP", ylim=(0, 1),
    )
    axes[0].legend()

    # 2. Recall of hidden positives
    for strategy, res in results.items():
        c = STRATEGY_COLORS.get(strategy, "#888888")
        axes[1].plot(budget, res["recall_mean"], label=strategy, color=c)
        axes[1].fill_between(
            budget,
            np.clip(res["recall_mean"] - res["recall_std"], 0, 1),
            res["recall_mean"] + res["recall_std"],
            alpha=0.2, color=c,
        )
    axes[1].set(
        title="Recall (hidden positives discovered)",
        xlabel="Labeled genes (budget)", ylabel="Recall",
    )
    axes[1].legend()

    # 3. AULC bar
    strategies = list(results.keys())
    aulc_vals  = [results[s]["aulc"] for s in strategies]
    bars = axes[2].bar(
        strategies, aulc_vals,
        color=[STRATEGY_COLORS.get(s, "#888888") for s in strategies],
        alpha=0.85,
    )
    for bar, v in zip(bars, aulc_vals):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.02 * max(aulc_vals),
            f"{v:.2f}", ha="center", fontsize=9,
        )
    axes[2].set(title="AULC (area under AP curve)\nhigher = faster learning", ylabel="AULC")

    # 4. Wasted queries
    rounds_q   = np.arange(1, N_ROUNDS + 1)
    random_exp = QUERY_SIZE * (n_unknown / len(universe))
    for strategy, res in results.items():
        c = STRATEGY_COLORS.get(strategy, "#888888")
        w = res["wasted_mean"]
        axes[3].plot(rounds_q[:len(w)], w, label=strategy, color=c)
    axes[3].axhline(
        random_exp, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
        label=f"random expected ({random_exp:.2f})",
    )
    axes[3].set(
        title="Wasted queries/round\n(unknown genes — not labeled)",
        xlabel="Round", ylabel="Wasted queries",
    )
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "learning_curves.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Plot -> {out}")
    plt.close()


# -- Main -------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Behavioural AL — Binary Classification")
    print(f"  Target    : {TARGET_COL} >= {THRESHOLD}")
    print(f"  Rounds    : {N_ROUNDS}  |  Query/round: {QUERY_SIZE}  |  Seed: {N_SEED}")
    print(f"  Trials    : {N_TRIALS}")
    print("=" * 60)

    print("\nLoading data ...")
    gene_df, feat_cols = load_gene_level_data(
        target_col=TARGET_COL, include_esm2=USE_ESM2, include_go=USE_GO,
    )

    universe = gene_df.copy().reset_index(drop=True)
    universe["label"] = (universe[TARGET_COL] >= THRESHOLD).astype(int)

    confirmed_mask = universe["has_annotation"].values > 0
    n_pos       = int(universe["label"].sum())
    n_confirmed = int(confirmed_mask.sum())
    n_unknown   = int((~confirmed_mask).sum())

    print(f"  Universe          : {len(universe):,} genes")
    print(f"  Features          : {len(feat_cols)}")
    print(f"  Positives         : {n_pos:,} ({100*n_pos/len(universe):.1f}%)")
    print(f"  Confirmed negative: {n_confirmed - n_pos:,}")
    print(f"  Unknown           : {n_unknown:,}")

    X = universe[feat_cols].values.astype(float)
    y = universe["label"].values.astype(int)

    results: dict[str, dict] = {}
    for strategy in ("random", "margin", "ucb"):
        results[strategy] = run_strategy(X, y, strategy, confirmed_mask)

    strategies = list(results.keys())
    print("\nPairwise t-tests on final AP (two-sided):")
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i + 1:]:
            t, p = stats.ttest_ind(
                results[s1]["final_ap_trials"],
                results[s2]["final_ap_trials"],
            )
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            print(f"  {s1} vs {s2}: t={t:.3f}  p={p:.4f}  {sig}")

    summary = pd.DataFrame([
        {
            "strategy":        s,
            "final_ap_mean":   res["ap_mean"][-1],
            "final_ap_std":    res["ap_std"][-1],
            "final_rec_mean":  res["recall_mean"][-1],
            "final_rec_std":   res["recall_std"][-1],
            "aulc":            res["aulc"],
        }
        for s, res in results.items()
    ])
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    for s, res in results.items():
        pd.DataFrame({
            "ap_mean":     res["ap_mean"],
            "ap_std":      res["ap_std"],
            "recall_mean": res["recall_mean"],
            "recall_std":  res["recall_std"],
        }).to_csv(RESULTS_DIR / f"curve_{s}.csv", index=False)

    print("\nSaving plots ...")
    plot_results(results, universe)

    print("\n" + "-" * 60)
    print(summary.to_string(index=False))
    print("-" * 60)
    print(f"\nOutputs -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
