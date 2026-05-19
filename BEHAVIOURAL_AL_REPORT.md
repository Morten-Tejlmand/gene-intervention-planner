# Behavioural Active Learning for C. elegans Gene Prioritisation

**Project:** Gene Intervention Planner — Active Learning Module  
**Date:** May 2026  
**Code:** `experiments/behavioural_al/`

---

## Overview

This document describes a pool-based active learning (AL) system designed to answer the question:

> **Can a model-guided experiment selection strategy find genes that produce behavioural phenotypes in *C. elegans* faster than random sampling, given a fixed experimental budget?**

The system frames gene prioritisation as a cyber-physical control problem: the model acts as a controller that reads phenotype annotations (sensors), selects genes to perturb (actions), and attempts to maximise biological knowledge under a limited number of experiments (budget constraint).

---

## Background: Why the Previous Setup Was Insufficient

The earlier experiment used `is_neural` — a binary flag indicating whether a gene had any neural annotation in WormBase — as the prediction target. This had two critical problems:

1. **Too few positives.** Only ~3% of genes qualified, making the class imbalance severe and the learning signal sparse.
2. **Wrong objective.** Knowing a gene is neural does not tell a researcher whether it is *worth experimenting on*. Many neural genes are already well-characterised. The research value lies in identifying genes that, when perturbed, produce a measurable behavioural change — particularly less-studied ones.

The new system replaces `is_neural` with a purpose-built target: `neural_behaviour_count`.

---

## Target Variable: `neural_behaviour_count`

### Construction

The target was built by parsing the raw WormBase phenotype annotation file:

```
data/raw/wormbase/caenorhabditis_elegans.PRJNA13758.WBPS19.phenotypes.gaf
```

This GAF (Gene Annotation Format) file contains one row per gene–phenotype association, where each phenotype is identified by a `WBPhenotype` ontology term ID.

A curated set of **230 WBPhenotype IDs** was assembled covering behavioural and neural categories relevant to *C. elegans* neuroscience:

| Category | Example phenotypes |
|---|---|
| Locomotion | paralysed, uncoordinated, slow movement, backward locomotion defect |
| Chemosensation | chemotaxis defective, olfactory adaptation defective |
| Synaptic transmission | aldicarb resistance, levamisole resistance, acetylcholine secretion |
| Pharyngeal pumping | pumping rate decreased, pharyngeal muscle contraction |
| Egg laying | egg laying defective, hyperactive egg laying |
| Defecation | defecation motor programme defective |
| Body wall muscle | muscle structure abnormal, myofilament disorganised |
| Paralysis / aggregation | protein aggregation increased, paralysed |

For each gene, `neural_behaviour_count` is the **number of distinct phenotype IDs from this set** that the gene has been annotated with. This is a literature count: a gene with count = 8 has been associated with 8 distinct neural behavioural phenotypes across published studies.

### Distribution

| Threshold | Positive genes | % of genome |
|---|---|---|
| count >= 1 | ~2,476 | ~12.4% |
| count >= 2 | ~1,200 | ~6.1% |

**Maximum count observed: ~50** (highly pleiotropic genes such as *unc-22*, *unc-32*).

The 12.4% positive rate at threshold = 1 is far more tractable for active learning than the previous 3%. It also better reflects the actual fraction of *C. elegans* genes known to have behavioural roles.

### Why this variable is appropriate

- **Biologically meaningful.** Each count represents an independent experimental observation of a behavioural effect. A high count means the gene's perturbation reliably produces phenotypes.
- **Actionable.** The research question "which gene should I perturb next?" is directly answered by high-count genes — they are most likely to produce detectable behavioural readouts.
- **Honest uncertainty.** Count = 0 is ambiguous: it means either the gene has no behavioural role, or it has never been tested. The confirmed/unknown distinction (see below) handles this explicitly.

> **Note on regression vs classification:** `neural_behaviour_count` is a literature count, not a physical effect size. A count of 8 does not mean "8× bigger effect" — it means the gene appeared in 8 papers. Binary classification (above/below threshold) is therefore more semantically honest than regression until real Tierpsy movement tracking data (speed, reversals, omega turns) is available.

---

## Features

All features are derived from gene sequence and orthology data — **no annotation leakage** into the feature set.

### Biophysical features (9)
Computed from protein sequence via Biopython:

| Feature | Description |
|---|---|
| `length` | Protein sequence length (amino acids) |
| `molecular_weight` | Molecular weight (Da) |
| `aromaticity` | Fraction of aromatic residues (F, W, Y) |
| `instability_index` | Guruprasad instability index |
| `isoelectric_point` | Theoretical pI |
| `gravy` | Grand average of hydropathicity (GRAVY) |
| `helix_fraction` | Secondary structure helix propensity |
| `turn_fraction` | Secondary structure turn propensity |
| `sheet_fraction` | Secondary structure sheet propensity |

### Amino acid composition (~20)
`percent_A`, `percent_C`, ..., `percent_Y` — fractional content of each amino acid.

### Human alignment score (1)
BLAST alignment identity score against the human proteome; proxy for evolutionary conservation.

### Ortholog features (5)

| Feature | Description |
|---|---|
| `feature_ortholog_count` | Total number of orthologous sequences |
| `feature_human_ortholog_count` | Number of human orthologs |
| `feature_max_query_identity` | Best BLAST identity to any ortholog |
| `vertebrate_ortholog_count` | Vertebrate-specific ortholog count |
| `ortholog_species_count` | Number of species with orthologs |

### Optional features (currently disabled)
- **ESM-2 embeddings** (640-dim): protein language model embeddings. Enabled via `USE_ESM2 = True`.
- **GO SVD features**: Gene Ontology term vectors compressed via SVD. Enabled via `USE_GO = True`.

Both are disabled by default to avoid annotation leakage risk and to keep experiments fast.

---

## Confirmed Positive / Negative Distinction

A core challenge in this dataset is the **Positive-Unlabelled (PU) learning problem**:

- Genes with `has_annotation > 0` have been studied in WormBase — their count is reliable.
  - `neural_behaviour_count >= THRESHOLD` → **Confirmed positive**
  - `neural_behaviour_count < THRESHOLD` → **Confirmed negative** (gene was studied, no neural behavioural effect found)
- Genes with `has_annotation == 0` have never appeared in WormBase — their count = 0 means *untested*, not *negative*.

### How this shapes the active learning loop

```
All genes
├── Confirmed positive  (neural_behaviour_count >= THRESHOLD)
├── Confirmed negative  (has_annotation > 0, count < THRESHOLD)  ← reliable y=0
└── Unknown             (has_annotation == 0)                     ← y=0 is unreliable
```

**Test set** is drawn only from confirmed genes (positives + confirmed negatives) and held out from the start. It is never queried.

**Pool** contains all remaining confirmed genes plus all unknown genes. When the model selects a query from the pool:
- If the gene is **confirmed** → it is added to the labeled training set.
- If the gene is **unknown** → it is removed from the pool but **not added to the labeled set**. This counts as a **wasted query** — the real-world equivalent of running an experiment on a gene that has no existing phenotype data to validate against.

This design means that intelligent acquisition strategies are penalised for wasting budget on unknown genes, which is tracked as a per-round metric.

### Threshold setting

The positive/negative boundary is controlled by:

```python
THRESHOLD = 1   # in run_experiment.py, line 71
```

| Threshold | Positive rate | Rationale |
|---|---|---|
| 1 | 12.4% | Any documented neural behavioural effect |
| 2 | ~6.1% | At least two independent studies — more reliable signal |

Threshold = 2 is more conservative but reduces noise from single-paper findings.

---

## Active Learning Loop

```
1. Seed: draw N_SEED=20 confirmed genes (pos + neg) at random
2. For each round (up to N_ROUNDS=25):
   a. Train EnsembleClassifier on labeled set
   b. Evaluate Average Precision on held-out test set
   c. Score all pool genes with acquisition function
   d. Query top QUERY_SIZE=10 genes
   e. Add confirmed queries to labeled set; discard unknown queries
3. Repeat for N_TRIALS=10 independent random seeds, report mean ± std
```

---

## Acquisition Strategies

Three strategies are compared:

### Random (baseline)
```python
scores = np.random.rand(len(pool))
```
Selects genes uniformly at random. Represents the cost of *not* using a model — the minimum bar the AL strategies must beat.

### Margin Sampling
```python
scores = 1.0 - np.abs(proba - 0.5) * 2
```
Selects genes where the model is most uncertain — those closest to the decision boundary (P(positive) ≈ 0.5). This is **pure exploration**: it finds genes that would most change the model's beliefs, but does not directly exploit genes already thought to be positive.

**Best for:** refining the decision boundary; useful when the goal is a well-calibrated classifier.

### Upper Confidence Bound (UCB)
```python
scores = proba + beta * std
```
Combines exploitation (`proba` — query genes likely to be positive) with exploration (`std` — query genes the ensemble disagrees on). Beta controls the balance; default `beta = 1.0`.

**Best for:** maximising recall of positives quickly; the natural choice when the goal is to find as many positive genes as possible within a fixed budget.

### Why these three?
- **Random** is the only honest baseline for AL: any strategy that doesn't beat it is useless.
- **Margin** is the most common AL strategy in the literature, providing a reference for pure uncertainty sampling.
- **UCB** is the theoretically motivated choice for this specific problem — we want to *find positive genes*, not just *build a good classifier*. The UCB formulation from multi-armed bandits directly optimises for this.

---

## Ensemble Model

The classifier is a three-member ensemble:

```
EnsembleClassifier
├── RandomForestClassifier       (n_estimators=50, class_weight='balanced')
├── GradientBoostingClassifier   (n_estimators=50, max_depth=3, lr=0.1)
└── ExtraTreesClassifier         (n_estimators=50, class_weight='balanced')
```

**Mean probability** across members is used as the acquisition score for UCB.  
**Standard deviation** across members is used as the uncertainty signal for both UCB and margin sampling.

`class_weight='balanced'` is applied to RF and ExtraTrees to compensate for the positive/negative imbalance without resampling.

Features are standardised with `StandardScaler` (fit on labeled set only, applied to pool and test set).

---

## Metrics

### Average Precision (primary)
Computed on a fixed held-out test set each round. AP summarises the precision-recall curve and is appropriate for imbalanced problems where recall of positives matters more than overall accuracy.

### Recall of hidden positives
Fraction of all true positives (not in the test set) that have been discovered in the labeled set. This directly measures how efficiently the strategy finds genes worth experimenting on.

### AULC (Area Under the Learning Curve)
Trapezoid integral of the mean AP curve across rounds. A single number summarising learning efficiency — higher means faster learning.

### Wasted queries per round
Number of unknown genes selected per round. A model that learns to avoid unknown genes saves experimental budget.

---

## Output Plots

Four panels are produced at `artifacts/behavioural_al/learning_curves.png`:

| Panel | Shows |
|---|---|
| Average Precision | AP on held-out test vs labeled budget — primary metric |
| Recall | Fraction of hidden positives found vs budget |
| AULC bar | Single-number efficiency comparison across strategies |
| Wasted queries | Unknown genes queried per round (lower = smarter model) |

The x-axis for learning curves is **labeled genes (budget)** = `N_SEED + round × QUERY_SIZE`, not round number. This makes strategies with different query sizes directly comparable.

---

## Configuration Reference

```python
# experiments/behavioural_al/run_experiment.py

TARGET_COL   = "neural_behaviour_count"
THRESHOLD    = 1      # count >= THRESHOLD → positive (try 1 or 2)
N_SEED       = 20     # confirmed genes in initial labeled set
N_ROUNDS     = 25     # AL rounds per trial
QUERY_SIZE   = 10     # genes queried per round
N_TRIALS     = 10     # independent random seeds for variance estimation
N_ESTIMATORS = 50     # trees per ensemble member
UCB_BETA     = 1.0    # exploration weight in UCB (higher = more exploration)

USE_ESM2     = False  # enable 640-dim ESM-2 protein embeddings
USE_GO       = False  # enable GO ontology SVD features
```

---

## File Structure

```
experiments/behavioural_al/
├── __init__.py
├── build_neural_target.py    # parses WormBase GAF → neural_behaviour_target.csv
├── data_loader.py            # loads features + merges neural target
├── models.py                 # EnsembleClassifier, EnsembleRegressor
├── acquisition.py            # margin_sampling, ucb_classification, random_acquisition
└── run_experiment.py         # main AL simulation loop + plots

data/processed/
├── enriched_v2_features.csv          # gene-level biophysical + ortholog features
├── neural_behaviour_target.csv       # built by build_neural_target.py
├── esm2_embeddings.csv               # optional protein language model embeddings
└── go_features.csv                   # optional GO SVD features

artifacts/behavioural_al/
├── learning_curves.png               # 4-panel output plot
├── summary.csv                       # final AP, recall, AULC per strategy
├── curve_random.csv                  # per-round AP + recall for random
├── curve_margin.csv                  # per-round AP + recall for margin
└── curve_ucb.csv                     # per-round AP + recall for UCB
```

---

## Limitations and Next Steps

| Limitation | Impact | Path forward |
|---|---|---|
| `neural_behaviour_count` is a literature count, not an effect size | High-count genes are well-studied, not necessarily high-effect | Integrate Tierpsy movement tracking data (speed, reversals, omega turns) |
| Selection bias: well-studied genes dominate labeled data | Model learns to prioritise already-known genes | Stratified sampling; density-weighted acquisition |
| Unknown genes are treated as wasted queries in simulation | In deployment, unknown genes are the primary targets | Transductive AL or PU learning methods |
| Features are sequence-only (no expression, network data) | Limited predictive power without functional context | Add tissue expression (single-cell RNA-seq), protein interaction network features |
| Threshold is arbitrary | Different questions need different positivity definitions | Threshold sensitivity analysis (1, 2, 3, 5) |
| 10 trials × 25 rounds is fast but noisy | Variance estimates may be wide | Increase N_TRIALS to 30–50 for publication |

---

## Running the Experiment

```bash
# From project root
python experiments/behavioural_al/build_neural_target.py   # only needed once
python experiments/behavioural_al/run_experiment.py
```

Results are written to `artifacts/behavioural_al/`.
