"""
Step 5 — Functional Feature Extraction
=======================================
Adds WormBase phenotype-annotation and ortholog-breadth features to every
protein isoform row, producing enriched_v2_features.csv.

New columns added per gene (replicated across isoforms):
  Phenotype  — annot_count, unique_pheno_count, positive_annot_count,
               positive_annot_rate, mean_evidence_weight, has_annotation,
               reference_count, neural_pheno_overlap
  Orthologs  — feature_ortholog_count, feature_human_ortholog_count,
               feature_max_query_identity, vertebrate_ortholog_count,
               ortholog_species_count

Why these features:
  * Phenotype annotations are direct experimental evidence — neural genes have
    locomotion, chemosensory, and synaptic phenotypes that sequence alone can't
    capture. neural_pheno_overlap measures how similar a gene's phenotype profile
    is to the 60 known neural seed genes.
  * Ortholog breadth captures how broadly conserved a gene is across the tree
    of life. Vertebrate conservation specifically flags mammalian-relevant genes.

Run from project root:
    python src/gene_intervention_planner/data/05_functional_features.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_PREFIX = "data/raw/wormbase/caenorhabditis_elegans.PRJNA13758.WBPS19"
GAF_PATH       = Path(f"{_PREFIX}.phenotypes.gaf")
ORTHOLOGS_PATH = Path(f"{_PREFIX}.orthologs.tsv")
BASE_CSV = Path("data/processed/enriched_genomic_features.csv")
OUT_CSV  = Path("data/processed/enriched_v2_features.csv")

# WBGene IDs of known neural-relevant seed genes (mirrors experiment constants)
# 60 genes covering synaptic, cholinergic, GABAergic, dopaminergic, serotonergic,
# glutamatergic, sensory, vesicle release, neuropeptide, and neural-TF systems
NEURAL_SEED_IDS: frozenset[str] = frozenset({
    # Original synaptic set
    "WBGene00003734", "WBGene00003816", "WBGene00006745", "WBGene00004354",
    "WBGene00004944", "WBGene00001183", "WBGene00001617", "WBGene00003681",
    "WBGene00000491", "WBGene00003152", "WBGene00006261", "WBGene00006259",
    "WBGene00001445", "WBGene00019756",
    # Cholinergic
    "WBGene00006756", "WBGene00000501", "WBGene00006765", "WBGene00002974",
    "WBGene00000042", "WBGene00000043",
    # GABAergic
    "WBGene00006762", "WBGene00006783", "WBGene00006784", "WBGene00001373",
    "WBGene00012915",
    # Dopaminergic
    "WBGene00000296", "WBGene00000934", "WBGene00001052", "WBGene00001053",
    "WBGene00020506", "WBGene00000295",
    # Serotonergic
    "WBGene00006600", "WBGene00003387", "WBGene00004776", "WBGene00004779",
    "WBGene00004780",
    # Glutamatergic (additional)
    "WBGene00001613", "WBGene00001615", "WBGene00001616",
    # Sensory / mechanosensory
    "WBGene00003174", "WBGene00003166", "WBGene00003167", "WBGene00003170",
    "WBGene00003889", "WBGene00003839", "WBGene00006616", "WBGene00006527",
    # Vesicle release machinery
    "WBGene00006757", "WBGene00006798", "WBGene00006364", "WBGene00000086",
    "WBGene00006767",
    # Neuropeptide processing
    "WBGene00001172", "WBGene00001189",
    # Axon development
    "WBGene00004457", "WBGene00001008",
    # Neural transcription factors
    "WBGene00006818", "WBGene00006654", "WBGene00000435",
    # Other neural
    "WBGene00000149",
})

VERTEBRATE_GENERA: frozenset[str] = frozenset({
    "Homo", "Mus", "Rattus", "Danio", "Xenopus", "Gallus", "Macaca",
    "Pan", "Bos", "Sus", "Canis", "Oryzias", "Takifugu", "Oncorhynchus",
    "Equus", "Ovis", "Monodelphis", "Ornithorhynchus",
})

EVIDENCE_WEIGHTS: dict[str, float] = {
    "IDA": 1.00, "IMP": 0.95, "IGI": 0.90, "IPI": 0.85,
    "IEP": 0.80, "TAS": 0.70, "NAS": 0.45, "IEA": 0.35,
}

NEW_COLS = [
    "annot_count", "unique_pheno_count", "positive_annot_count",
    "positive_annot_rate", "mean_evidence_weight", "has_annotation",
    "reference_count", "neural_pheno_overlap",
    "feature_ortholog_count", "feature_human_ortholog_count",
    "feature_max_query_identity", "vertebrate_ortholog_count",
    "ortholog_species_count",
]

# GAF column names (WormBase phenotype GAF, tab-separated, no header, ! = comment)
_GAF_COLS = [
    "db", "gene_id", "gene_symbol", "qualifier", "phenotype_id",
    "reference", "evidence_code", "with_from", "aspect",
    "db_object_name", "db_object_synonym", "db_object_type",
    "taxon", "date", "assigned_by", "annotation_extension", "gene_product_form",
]


def _read_gaf(path: Path) -> pd.DataFrame:
    """Read a WormBase phenotype GAF file into a pandas DataFrame."""
    rows = []
    n_cols = len(_GAF_COLS)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("!"):
                continue
            parts = line.split("\t")
            # Pad short rows; truncate long rows
            parts += [""] * (n_cols - len(parts))
            rows.append(parts[:n_cols])
    df = pd.DataFrame(rows, columns=_GAF_COLS)
    return df


def build_phenotype_features(gaf_path: Path) -> pd.DataFrame:
    """Per-gene phenotype summary from WormBase GAF.

    neural_pheno_overlap uses leave-one-out for seed genes: each seed gene's
    overlap is counted against phenotype IDs from the OTHER 59 seeds only,
    breaking the circularity where a gene defines its own reference set.
    Non-seed genes use the full 60-gene reference (they never contributed to it).
    """
    print(f"  Parsing {gaf_path.name} ...")
    raw = _read_gaf(gaf_path)

    raw["is_not"] = raw["qualifier"].str.upper().str.contains("NOT").fillna(False)
    raw["evidence_weight"] = raw["evidence_code"].map(EVIDENCE_WEIGHTS).fillna(0.55)
    raw["label"] = (~raw["is_not"]).astype(int)

    # Per-seed-gene phenotype ID sets (positive annotations only)
    seed_rows = raw[raw["gene_id"].isin(NEURAL_SEED_IDS) & ~raw["is_not"]]
    seed_gene_phenos: dict[str, set[str]] = (
        seed_rows.groupby("gene_id")["phenotype_id"]
        .apply(lambda s: set(s.dropna()))
        .to_dict()
    )

    # Full reference: union across all 60 seeds
    all_neural_pheno_ids: set[str] = set().union(*seed_gene_phenos.values())
    print(f"  Neural seed phenotype IDs discovered: {len(all_neural_pheno_ids)}")

    # Compute base aggregates using full reference set
    raw["is_neural_pheno"] = raw["phenotype_id"].isin(all_neural_pheno_ids).astype(float)

    agg = (
        raw.groupby("gene_id")
        .agg(
            annot_count=("phenotype_id", "count"),
            unique_pheno_count=("phenotype_id", "nunique"),
            positive_annot_count=("label", "sum"),
            mean_evidence_weight=("evidence_weight", "mean"),
            reference_count=("reference", "nunique"),
            neural_pheno_overlap=("is_neural_pheno", "sum"),
        )
        .reset_index()
    )

    # Leave-one-out correction for seed genes: recount overlap against the
    # reference built from all OTHER seeds (removes self-contribution).
    gene_phenos_all: dict[str, set[str]] = (
        raw.groupby("gene_id")["phenotype_id"]
        .apply(lambda s: set(s.dropna()))
        .to_dict()
    )
    for gene_id, own_phenos in seed_gene_phenos.items():
        loo_ref = set().union(
            *(phenos for gid, phenos in seed_gene_phenos.items() if gid != gene_id)
        )
        overlap = float(len(gene_phenos_all.get(gene_id, set()) & loo_ref))
        agg.loc[agg["gene_id"] == gene_id, "neural_pheno_overlap"] = overlap

    agg["positive_annot_rate"] = agg["positive_annot_count"] / agg["annot_count"].clip(lower=1)
    agg["has_annotation"] = 1.0
    return agg


def build_ortholog_features(orthologs_path: Path) -> pd.DataFrame:
    """Per-gene ortholog breadth from WormBase orthologs TSV."""
    print(f"  Parsing {orthologs_path.name} ...")

    ort = pd.read_csv(
        orthologs_path,
        sep="\t",
        usecols=["gene_id", "ortholog_species_name", "query_identity"],
        low_memory=False,
    )
    ort["query_identity"] = pd.to_numeric(ort["query_identity"], errors="coerce")
    first_word = ort["ortholog_species_name"].str.split().str[0].fillna("")
    ort["is_human"] = (ort["ortholog_species_name"] == "Homo sapiens").astype(float)
    ort["is_vertebrate"] = first_word.isin(VERTEBRATE_GENERA).astype(float)

    agg = (
        ort.groupby("gene_id")
        .agg(
            feature_ortholog_count=("ortholog_species_name", "count"),
            feature_human_ortholog_count=("is_human", "sum"),
            feature_max_query_identity=("query_identity", "max"),
            vertebrate_ortholog_count=("is_vertebrate", "sum"),
            ortholog_species_count=("ortholog_species_name", "nunique"),
        )
        .reset_index()
    )
    agg["feature_max_query_identity"] = agg["feature_max_query_identity"].fillna(0.0)
    return agg


def main() -> None:
    print("=" * 60)
    print("Step 5: Functional Feature Extraction")
    print("=" * 60)

    base = pd.read_csv(BASE_CSV)
    print(f"  Base: {base.shape[1]} features × {len(base):,} isoforms")

    pheno_feats = build_phenotype_features(GAF_PATH)
    ort_feats   = build_ortholog_features(ORTHOLOGS_PATH)

    # Join on WBGene ID: common_name in base == gene_id in annotation tables
    result = (
        base
        .merge(pheno_feats.rename(columns={"gene_id": "common_name"}),
               on="common_name", how="left")
        .merge(ort_feats.rename(columns={"gene_id": "common_name"}),
               on="common_name", how="left")
    )

    for col in NEW_COLS:
        if col in result.columns:
            result[col] = result[col].fillna(0.0)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)

    n_new = result.shape[1] - base.shape[1]
    print(f"\n  Added {n_new} new feature columns")
    print(f"  Output: {OUT_CSV}")
    print(f"  Shape: {result.shape[1]} cols × {len(result):,} rows")

    coverage = (
        result.loc[result["common_name"].isin(NEURAL_SEED_IDS),
                   ["common_name", "annot_count", "neural_pheno_overlap",
                    "feature_ortholog_count", "vertebrate_ortholog_count"]]
        .drop_duplicates("common_name")
        .sort_values("annot_count", ascending=False)
    )
    print("\n  Neural seed gene annotation coverage:")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
