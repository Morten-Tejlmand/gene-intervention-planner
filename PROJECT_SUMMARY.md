# Gene Intervention Planner — Project Summary

## Purpose

The project builds an **active learning system** to identify which *C. elegans* genes are neurally relevant, without having to experimentally test every gene. Instead of random testing, the system trains a model on a small set of known neural genes and strategically selects which unknown genes to test next, maximizing discoveries per experiment. Experiments 1–7 form a progression: first proving the idea works, then comparing increasingly sophisticated selection strategies.

---

## Data Pipeline

### `src/.../01_dataset_all.py` — Protein Feature Extraction

Reads protein sequences from a WormBase FASTA file and computes **biophysical features** for every protein isoform using BioPython's `ProteinAnalysis`:

- **Sequence-level**: length, molecular weight, aromaticity, instability index, isoelectric point, GRAVY score (hydrophobicity)
- **Secondary structure**: predicted fractions of helix, turn, and sheet
- **Amino acid composition**: percentage of each of the 20 amino acids

**Why these features?** They can be computed directly from the protein sequence without running any wet-lab experiment. They serve as the base representation that the models learn from.

---

### `src/.../02_label_genes.py` — Seed Labeling

Takes the feature table and stamps a binary label (`is_neural = 1`) onto a small set of well-known neural genes, initially just 5: *nlg-1, nrx-1, unc-13, ric-4, snb-1*. Everything else gets label 0.

**Why so few?** This is intentional — it mimics the real-world situation where only a handful of genes have been experimentally confirmed. Active learning is only useful when labeled data is scarce.

---

### `src/.../03_ortholog_mapper.py` — Ortholog Conservation Score

Scores each gene by how well it aligns to a human ortholog. This produces a `human_alignment_score` column.

**Why?** Genes conserved between worms and humans are more likely to have important, shared biological functions. Conservation acts as an evolutionary signal that helps distinguish functionally important genes from background.

---

### `src/.../04_synptic_mapping.py` — Synaptic Network Mapping

Maps genes onto the known synaptic gene network. Genes connected to established synaptic components are more likely to themselves have synaptic/neural functions.

---

### `src/.../05_functional_features.py` — WormBase Functional Features

The most important enrichment step. Reads two official WormBase files — a phenotype annotation file (GAF format) and an ortholog table — and adds functional features to each gene.

**Phenotype features:**
- `annot_count` — total number of phenotype annotations
- `unique_pheno_count` — number of distinct phenotypes observed
- `positive_annot_count` / `positive_annot_rate` — how many annotations are "positive" (i.e., a phenotype was observed)
- `mean_evidence_weight` — quality/reliability of the evidence
- `has_annotation` — binary: any annotation at all?
- `reference_count` — number of literature references supporting annotations
- `neural_pheno_overlap` — how similar the gene's phenotype profile is to the 60 known neural seed genes

**Ortholog breadth features:**
- `feature_ortholog_count` — total number of species with an ortholog
- `feature_human_ortholog_count` — does it have a human ortholog?
- `feature_max_query_identity` — how similar the closest ortholog is
- `vertebrate_ortholog_count` — orthologs in vertebrates specifically
- `ortholog_species_count` — breadth of conservation across the tree of life

**Why?** Protein sequence alone cannot tell you if a gene causes locomotion defects or chemosensory failure. Phenotype annotations are direct experimental evidence. The `neural_pheno_overlap` feature is particularly powerful because it directly measures similarity to the known positive class. The output file `enriched_v2_features.csv` is what all experiments use.

---

## Experiments

### `exp01_baseline_neural_relevance.py` — Baseline Validation

**Purpose:** Prove that the features contain enough signal to predict neural relevance before attempting active learning.

**Models:** Logistic Regression, Random Forest, Gradient Boosting — all with `class_weight="balanced"` to handle the extreme class imbalance (~0.3% positive rate).

**Evaluation:** 5-fold stratified cross-validation. Metrics: AUROC, F1, Average Precision (AP).

**Why Average Precision as primary metric?** When only 0.3% of genes are positive, AUROC can be misleading (a model that ranks positives slightly above negatives scores well). AP penalizes the model for positives that appear low in the ranking, making it appropriate for rare-class detection.

**Why these three models?** They represent a range of complexity — Logistic Regression is interpretable and fast, Random Forest handles non-linear interactions, Gradient Boosting is typically the strongest baseline. If even LR works, the signal is real.

---

### `exp02_feature_comparison.py` — Feature Ablation

**Purpose:** Determine which feature groups contribute to predictive performance, so later experiments use only what is necessary.

**Model:** Random Forest (class-weighted), 5-fold CV.

**Feature sets tested (cumulative):**

| Set | Contents |
|-----|----------|
| A | Biophysics only (length, MW, charge, hydrophobicity, secondary structure) |
| B | A + amino acid percentages (20 AA frequencies) |
| C | B + conservation (human alignment score) |
| D | C + WormBase functional features (phenotype annotations, ortholog breadth) |

**Why cumulative sets?** To isolate the marginal gain from each group. If Set C is barely better than Set B, then conservation is not worth the complexity. If Set D jumps significantly over Set C, the functional features justify the extra data pipeline steps.

---

### `exp03_al_vs_random.py` — Core Active Learning vs Random

**Purpose:** The central experiment. Does model-guided gene selection find more neural genes than random selection, given the same experiment budget?

**Model:** Logistic Regression (C=1.0, class_weight="balanced"). LR is chosen here because it is fast and gives well-calibrated probabilities, which uncertainty sampling depends on.

**Simulation setup:**
- Start with 8 known neural genes + 40 negatives as the labeled seed
- Hold out 4 neural genes + 200 negatives for evaluation (never queried)
- Remaining genes form the unlabeled pool, including hidden neural genes the oracle reveals when queried
- Run 20 rounds, querying 15 genes per round
- Repeat 10 independent trials to get error bars

**Acquisition strategies:**

| Strategy | Description |
|----------|-------------|
| Random | Uniform random selection from the pool (baseline) |
| Uncertainty (margin) | Selects genes where the model is least confident: `score = 1 - \|P(neural) - P(non-neural)\|`. Small margin = high uncertainty = most informative. |
| Query-by-committee (QBC) | Trains 3 LR models with different regularization strengths (C = 0.1, 1, 10) and selects genes where their predictions disagree most (highest variance). |

**Why QBC?** If a single model is miscalibrated, the committee catches it. Disagreement among models signals that a gene sits near a decision boundary none of them has fully resolved.

**Metrics:** Average Precision on the held-out test set (primary), recall of hidden neural genes discovered through querying (secondary).

---

### `exp04_hybrid_acquisition.py` — Hybrid Acquisition

**Purpose:** Pure uncertainty sampling finds genes the model is confused about — but those might be biologically uninteresting. This experiment tests strategies that combine uncertainty with biological relevance.

**Model:** Logistic Regression (same as Exp 3).

**Acquisition strategies:**

| Strategy | Description |
|----------|-------------|
| Uncertainty | Margin sampling — same as Exp 3, acts as reference |
| Neural score | Selects genes with highest predicted P(neural=1). Pure exploitation: trust the model and mine its top predictions. |
| Hybrid | Weighted combination: 40% uncertainty + 40% neural probability + 20% conservation score, all normalized to [0,1] |

**Why the hybrid?** Uncertainty alone can waste queries on genes that are hard to classify but have low neural probability. Neural score alone may cluster queries around one functional group, missing diversity. The hybrid balances exploration and exploitation, with conservation as a tiebreaker grounded in biology.

**Extra metric:** Enrichment — what proportion of queried genes turn out to be neural. This measures practical lab efficiency: if you're running 15 experiments per round, how many will yield a positive result?

---

### `exp05_interaction_prioritization.py` — Gene Pair Prioritization

**Purpose:** Extend from single-gene selection to **pairwise combinations**. Two genes can interact — disrupting both simultaneously can have a different effect than disrupting either alone (epistasis). This experiment ranks which gene pairs are most worth testing.

**Model:** Random Forest trained on the full labeled set. The top 60 predicted neural genes form the candidate pool for pairing.

**Composite pair score (weights are tunable):**

| Component | Weight | Calculation |
|-----------|--------|-------------|
| Model relevance | 35% | Mean of the two genes' predicted neural probabilities |
| Hydrophobic fit | 15% | `1 / (1 + \|GRAVY_A - GRAVY_B\|)` — similar hydrophobicity suggests proteins act in the same environment |
| Electrostatic drive | 20% | `\|pI_A - pI_B\|` — large charge difference promotes protein-protein attraction |
| Joint stability | 15% | `1 / (1 + instability_A × instability_B / 1000)` — both proteins should be stable enough to study |
| Conservation match | 15% | Mean human alignment score — prioritize conserved pairs |

**Why score pairs?** The combinatorial space is enormous. This shows how the model can reduce that space to a manageable shortlist before committing lab resources.

**Evaluation:** Top-K hit rate — what fraction of the top-ranked pairs contain at least one known synaptic anchor gene (nlg-1, nrx-1, shn-1, unc-13, ric-4, snb-1).

---

### `exp06_pu_learning.py` — Positive-Unlabeled (PU) Learning

**Purpose:** Address a fundamental problem in the data: genes labeled 0 (non-neural) are not *confirmed* negatives — they are simply *untested*. Some of them are probably neural. A standard model that treats all unlabeled genes as true negatives will be biased, placing the decision boundary in the wrong place.

**Solution — PU Bagging (Mordelet & Vert, 2014):**
- Train 15 Random Forest classifiers (bags)
- Each bag gets: all confirmed positive genes + a random subsample of unlabeled genes (treated as negatives for that bag, 20× the number of positives)
- Average predicted P(neural=1) across all 15 bags
- Because each bag sees a *different* random subset of unlabeled genes, the ensemble averages out noise from incorrectly-labeled negatives

**Strategies compared:** Both use uncertainty-sampling (margin-based) acquisition:

| Strategy | Description |
|----------|-------------|
| Standard RF | Single Random Forest with `class_weight="balanced"` |
| PU Bagging | 15-bag ensemble as described above |

**Why compare them under the same acquisition function?** To isolate whether the model type matters, independent of the selection strategy. Any performance difference comes purely from how each model handles unlabeled genes.

---

### `exp07_advanced_acquisition.py` — Advanced Acquisition Strategies

**Purpose:** Push beyond the strategies in Exp 3–4, using the Random Forest ensemble's internal tree structure to compute more principled acquisition scores. All strategies use the same RF (50 trees, `class_weight="balanced"`) so differences in performance come only from the acquisition function.

**Acquisition strategies:**

| Strategy | Description |
|----------|-------------|
| Random | Uniform baseline |
| Neural score | Highest mean P(neural) across trees. Pure exploitation. Replicates Exp 4 but with RF. |
| BALD with exploitation bias | 40% BALD mutual information (spread of predictions across trees) + 60% mean predicted probability. Finds genes that are both informative and likely neural. |
| Thompson sampling | For each of the 15 queries in a batch, randomly sample one tree and pick the highest-scoring pool gene under that tree. Naturally balances exploration and exploitation — each tree represents a plausible model of the world. |
| Diverse exploitation | Takes the top 10% of pool genes by predicted neural probability, then runs greedy farthest-first selection in feature space to pick 15 maximally spread-out candidates. Prevents the batch from clustering around one functional group. |
| Boundary exploration | Score = `P(neural) × distance to nearest known positive` in feature space. Prioritizes high-probability genes that are *far* from already-known positives, actively seeking new positive clusters. |

**Metrics:** Average Precision on the held-out test set (primary), recall of hidden neural genes discovered (secondary).

---

### `experiments/analyse_results.py` — Results Aggregator

Loads the `summary.csv` output files from all 7 experiments and prints a unified comparison report. No modeling or new computations — purely aggregation for easy side-by-side comparison of all strategies after all experiments have run.

---

## Shared Design Decisions

| Decision | Reason |
|----------|--------|
| 10 trials per experiment | Each trial uses a different random seed. Averaging over 10 gives reliable estimates and meaningful error bars, so we can tell whether one strategy is genuinely better or just lucky on a particular split. |
| `class_weight="balanced"` on all models | The positive class is ~0.3% of the dataset. Without weighting, a model that always predicts "non-neural" achieves 99.7% accuracy. Weighting forces the model to care about rare positives. |
| Fixed held-out test set | To measure generalization. If test genes were in the pool, the model could memorize them during training. The held-out set measures whether the model has learned something transferable, not just memorized training examples. |
| Average Precision over AUROC | At 0.3% class imbalance, AUROC inflates. AP rewards models that rank positives at the very top of the list — exactly what you want when selecting which genes to test next with a limited lab budget. |
| Oracle simulation | Hidden positive genes are placed in the unlabeled pool. When the model selects them, their true label is "revealed". This simulates the real-world loop where you run the experiment and get a result. |
