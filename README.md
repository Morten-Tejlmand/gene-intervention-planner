# Gene Intervention Planner

Reproducible Python project for active learning of next-best genome experiment candidates using WormBase-style mock data and processed WormBase annotations.

## Stack

- Python 3.12
- `uv` dependency management
- `polars` for tabular processing
- `scikit-learn` for baseline modeling
- `numpy` for numerical operations

## Project Layout

```text
configs/            Experiment configuration files
data/raw/           Mock WormBase-style input files (CSV + Parquet)
data/processed/     Optional processed intermediate datasets
artifacts/          Generated ranking outputs and run artifacts
src/                Package source code
tests/              Unit and integration tests
```

## Setup

```bash
uv sync --all-groups
```

## Generate Mock Data

```bash
uv run gene-planner generate-mock-data --config configs/experiment.yaml
```

This writes:

- `data/raw/wormbase_mock_genes.csv`
- `data/raw/wormbase_mock_phenotypes.csv`
- `data/raw/wormbase_mock_candidates.parquet`

## Run Full Active Learning Experiment

```bash
uv run gene-planner run --config configs/experiment.yaml
```

Outputs are written under `artifacts/<run_id>/`:

- `ranking.csv` and `ranking.parquet`
- `round_metrics.csv`
- `acquisition_log.csv`
- `run_summary.json`

## Run Tests

```bash
uv run pytest
```

## Notes

- Default task is binary hit prediction (`label` in `{0,1}`).
- Simulator mode reveals hidden truth labels for selected unlabeled candidates with optional noise.
- Data source interfaces are designed so a real WormBase fetcher can be added later without changing pipeline orchestration.
- `evaluate_predictions(...)` in `src/gene_intervention_planner/modeling/active_learning.py` is the simple metric function used for validation.

## Build `gap-2` Candidate Table From WormBase Processed Data

```bash
uv run gene-planner build-gap2-dataset \
  --phenotypes-gaf data/raw/wormbase/caenorhabditis_elegans.PRJNA13758.WBPS19.phenotypes.gaf \
  --orthologs-tsv data/raw/wormbase/caenorhabditis_elegans.PRJNA13758.WBPS19.orthologs.tsv \
  --focus-gene gap-2 \
  --comparator-gene nlg-1 \
  --comparator-gene nrx-1 \
  --output data/processed/wormbase_gap2_candidates.parquet
```

Then run the active-learning experiment on that real candidate table:

```bash
uv run gene-planner run --config configs/gap2_experiment.yaml
```

## Download + Preprocess Real WormBase Data

This command downloads true WormBase release files, preprocesses them, and writes outputs under `data/processed/`:

```bash
uv run gene-planner prepare-wormbase-data \
  --raw-dir data/raw/wormbase \
  --processed-dir data/processed \
  --focus-gene gap-2 \
  --comparator-gene nlg-1 \
  --comparator-gene nrx-1
```

If raw files already exist locally, skip downloads and only preprocess:

```bash
uv run gene-planner prepare-wormbase-data --skip-download
```
