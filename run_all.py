"""
run_all.py — Full pipeline runner
==================================
Runs data processing → all experiments → results analysis.

Usage:
    python run_all.py               # run everything
    python run_all.py --skip-data   # skip data processing (use existing CSVs)
    python run_all.py --skip-slow   # skip steps 01 and 03 (the slow ones: ~5-15 min each)

Steps:
    Data pipeline:
        01  protein.fa  → universal_protein_features.csv    (slow: BioPython parsing)
        02  ↑           → labeled_genome_dataset.csv
        03  ↑           → enriched_genomic_features.csv     (slow: pairwise alignment)
        05  ↑ + GAF + orthologs → enriched_v2_features.csv  (fast)

    Experiments (all use enriched_v2_features.csv):
        exp01  baseline model comparison
        exp02  feature set comparison
        exp03  active learning vs random
        exp04  hybrid acquisition strategies
        exp05  gene pair prioritization
        exp06  PU bagging vs standard RF                    (slow: large ensemble)

    Analysis:
        analyse_results.py  → unified results table
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
PROCESSED = ROOT / "data" / "processed"


def run(label: str, cmd: list[str], required_output: Path | None = None) -> bool:
    """Run a subprocess step. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if required_output and required_output.exists():
        print(f"  [SKIP] Output already exists: {required_output.name}")
        print(f"  (delete it to force re-run)")
        return True

    t0 = time.time()
    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONWARNINGS": "ignore"},
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  [FAILED] {label} (exit code {result.returncode})")
        return False

    print(f"\n  [OK] {label} — {elapsed:.1f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip all data processing steps (use existing CSVs)")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip steps 01 and 03 (protein parsing + alignment)")
    parser.add_argument("--skip-exp06", action="store_true",
                        help="Skip exp06 (PU bagging — very slow)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Gene Intervention Planner — Full Pipeline")
    print("=" * 60)

    failed: list[str] = []

    # ------------------------------------------------------------------
    # Data pipeline
    # ------------------------------------------------------------------
    if not args.skip_data:
        steps = [
            (
                "Step 01: Protein feature extraction (FASTA → CSV)",
                ["src/gene_intervention_planner/data/01_dataset_all.py"],
                PROCESSED / "universal_protein_features.csv",
                args.skip_slow,
            ),
            (
                "Step 02: Gene labeling (seed WBGene IDs)",
                ["src/gene_intervention_planner/data/02_label_genes.py"],
                PROCESSED / "labeled_genome_dataset.csv",
                False,
            ),
            (
                "Step 03: Human ortholog alignment scores",
                ["src/gene_intervention_planner/data/03_ortholog_mapper.py"],
                PROCESSED / "enriched_genomic_features.csv",
                args.skip_slow,
            ),
            (
                "Step 05: WormBase functional features (LOO-corrected)",
                ["src/gene_intervention_planner/data/05_functional_features.py"],
                None,   # always re-run so LOO correction is fresh
                False,
            ),
        ]

        for label, cmd, output, skip_if_slow in steps:
            if skip_if_slow and output and output.exists():
                print(f"\n  [SKIP --skip-slow] {label}")
                continue
            if not run(label, cmd, required_output=None if skip_if_slow else output):
                failed.append(label)
                print("\n  Pipeline aborted: data step failed.")
                sys.exit(1)

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------
    experiments = [
        ("Exp 01: Baseline model comparison",        "experiments/exp01_baseline_neural_relevance.py"),
        ("Exp 02: Feature set comparison",           "experiments/exp02_feature_comparison.py"),
        ("Exp 03: Active learning vs random",        "experiments/exp03_al_vs_random.py"),
        ("Exp 04: Hybrid acquisition strategies",    "experiments/exp04_hybrid_acquisition.py"),
        ("Exp 05: Gene pair prioritization",         "experiments/exp05_interaction_prioritization.py"),
        ("Exp 06: PU Bagging vs standard RF",        "experiments/exp06_pu_learning.py"),
    ]

    for label, script in experiments:
        if args.skip_exp06 and "exp06" in script:
            print(f"\n  [SKIP --skip-exp06] {label}")
            continue
        if not run(label, [script]):
            failed.append(label)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    run("Results analysis", ["experiments/analyse_results.py"])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if failed:
        print(f"  DONE with {len(failed)} failure(s):")
        for f in failed:
            print(f"    - {f}")
    else:
        print("  ALL STEPS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
