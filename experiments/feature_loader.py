"""
Shared feature loading for all experiments.

Merges base biophysical/sequence features with ESM-2 embeddings and GO term
SVD features. Annotation-based columns (annot_count etc.) are kept in the
returned DataFrame for diagnostic use but are NEVER included in feat_cols,
preventing literature-popularity bias from leaking into the model.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/processed/enriched_v2_features.csv")
ESM2_PATH = Path("data/processed/esm2_embeddings.csv")
GO_PATH   = Path("data/processed/go_features.csv")

BIOPHYS_FEATURES = [
    "length", "molecular_weight", "aromaticity", "instability_index",
    "isoelectric_point", "gravy", "helix_fraction", "turn_fraction", "sheet_fraction",
]

ORTHOLOG_FEATURES = [
    "feature_ortholog_count", "feature_human_ortholog_count",
    "feature_max_query_identity", "vertebrate_ortholog_count",
    "ortholog_species_count",
]

# Retained in gene_df for diagnostic plots (literature-bias check) but never
# passed to any ML model as features.
ANNOTATION_COLS = [
    "annot_count", "unique_pheno_count", "positive_annot_count",
    "positive_annot_rate", "mean_evidence_weight", "has_annotation",
    "reference_count", "neural_pheno_overlap",
]


def load_gene_level_data(
    neural_gene_ids: dict[str, str],
    include_esm2: bool = True,
    include_go: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load, aggregate, and enrich gene-level feature data.

    Parameters
    ----------
    neural_gene_ids : dict mapping WBGene ID -> gene symbol for positive class
    include_esm2    : merge ESM-2 embeddings if the cache file exists
    include_go      : merge GO SVD features if the cache file exists

    Returns
    -------
    gene_df  : one row per gene; contains annotation cols for diagnostics
    feat_cols: clean feature columns to use as ML inputs (no annotation bias)
    """
    df = pd.read_csv(DATA_PATH)
    df["is_neural"] = df["common_name"].isin(neural_gene_ids).astype(int)

    aa_cols = sorted(c for c in df.columns if c.startswith("percent_"))

    all_base_cols = (
        BIOPHYS_FEATURES
        + ["human_alignment_score"]
        + aa_cols
        + ORTHOLOG_FEATURES
        + ANNOTATION_COLS
    )
    base_cols = [c for c in all_base_cols if c in df.columns]

    agg: dict = {c: "mean" for c in base_cols}
    agg["is_neural"] = "max"
    gene_df = df.groupby("common_name").agg(agg).reset_index()

    feat_cols = [
        c for c in
        BIOPHYS_FEATURES + ["human_alignment_score"] + aa_cols + ORTHOLOG_FEATURES
        if c in gene_df.columns
    ]

    if include_esm2:
        if ESM2_PATH.exists():
            esm2 = pd.read_csv(ESM2_PATH)
            esm2_cols = [c for c in esm2.columns if c.startswith("esm2_")]
            gene_df = gene_df.merge(
                esm2[["common_name"] + esm2_cols], on="common_name", how="left"
            )
            gene_df[esm2_cols] = gene_df[esm2_cols].fillna(0.0)
            feat_cols += esm2_cols
            print(f"  ESM-2  : {len(esm2_cols)} features merged")
        else:
            print(f"  ESM-2  : {ESM2_PATH} not found, skipping")

    if include_go:
        if GO_PATH.exists():
            go = pd.read_csv(GO_PATH)
            go_cols = [c for c in go.columns if c.startswith("go_svd_")]
            gene_df = gene_df.merge(
                go[["common_name"] + go_cols], on="common_name", how="left"
            )
            gene_df[go_cols] = gene_df[go_cols].fillna(0.0)
            feat_cols += go_cols
            print(f"  GO SVD : {len(go_cols)} features merged")
        else:
            print(f"  GO     : {GO_PATH} not found, skipping")

    return gene_df, feat_cols
