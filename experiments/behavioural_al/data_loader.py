# data loading for the behavioural AL experiments

import os
import pandas as pd
import numpy as np

DATA_PATH = "data/processed/enriched_v2_features.csv"
ESM2_PATH = "data/processed/esm2_embeddings.csv"
GO_PATH = "data/processed/go_features.csv"
NEURAL_TARGET_PATH = "data/processed/neural_behaviour_target.csv"

# basic biophysical protein properties
BIOPHYS_FEATURES = [
    "length", "molecular_weight", "aromaticity", "instability_index",
    "isoelectric_point", "gravy", "helix_fraction", "turn_fraction", "sheet_fraction",
]

# evolutionary features from ortholog analysis
ORTHOLOG_FEATURES = [
    "feature_ortholog_count", "feature_human_ortholog_count",
    "feature_max_query_identity", "vertebrate_ortholog_count",
    "ortholog_species_count",
]

# annotation-derived columns - DO NOT USE as features (data leakage)
ANNOTATION_COLS = [
    "annot_count", "unique_pheno_count", "positive_annot_count",
    "positive_annot_rate", "mean_evidence_weight", "has_annotation",
    "reference_count", "neural_pheno_overlap",
]


def load_gene_level_data(target_col="unique_pheno_count", include_esm2=True, include_go=True):
    """Load and aggregate features to gene level. Returns (gene_df, feat_cols)."""
    df = pd.read_csv(DATA_PATH)

    aa_cols = sorted([c for c in df.columns if c.startswith("percent_")])

    want_cols = BIOPHYS_FEATURES + ["human_alignment_score"] + aa_cols + ORTHOLOG_FEATURES + ANNOTATION_COLS
    base_cols = [c for c in want_cols if c in df.columns]

    # some genes have multiple rows - average them
    gene_df = df.groupby("common_name").agg({c: "mean" for c in base_cols}).reset_index()

    # feature columns = no annotation columns (would be data leakage)
    feat_cols = [c for c in BIOPHYS_FEATURES + ["human_alignment_score"] + aa_cols + ORTHOLOG_FEATURES
                 if c in gene_df.columns]

    if include_esm2:
        if os.path.exists(ESM2_PATH):
            esm2_df = pd.read_csv(ESM2_PATH)
            esm2_cols = [c for c in esm2_df.columns if c.startswith("esm2_")]
            gene_df = gene_df.merge(esm2_df[["common_name"] + esm2_cols], on="common_name", how="left")
            gene_df[esm2_cols] = gene_df[esm2_cols].fillna(0.0)
            feat_cols = feat_cols + esm2_cols
            print(f"  ESM-2 : {len(esm2_cols)} features merged")
        else:
            print(f"  ESM-2 : file not found ({ESM2_PATH}), skipping")

    if include_go:
        if os.path.exists(GO_PATH):
            go_df = pd.read_csv(GO_PATH)
            go_cols = [c for c in go_df.columns if c.startswith("go_svd_")]
            gene_df = gene_df.merge(go_df[["common_name"] + go_cols], on="common_name", how="left")
            gene_df[go_cols] = gene_df[go_cols].fillna(0.0)
            feat_cols = feat_cols + go_cols
            print(f"  GO SVD: {len(go_cols)} features merged")
        else:
            print(f"  GO    : file not found ({GO_PATH}), skipping")

    gene_df[feat_cols] = gene_df[feat_cols].fillna(0.0)

    if target_col == "neural_behaviour_count":
        if not os.path.exists(NEURAL_TARGET_PATH):
            raise FileNotFoundError(
                f"Neural target not found: {NEURAL_TARGET_PATH}\n"
                "Run build_neural_target.py first"
            )
        neural_df = pd.read_csv(NEURAL_TARGET_PATH)[["common_name", "neural_behaviour_count"]]
        gene_df = gene_df.merge(neural_df, on="common_name", how="left")
        gene_df["neural_behaviour_count"] = gene_df["neural_behaviour_count"].fillna(0.0)
    elif target_col not in gene_df.columns:
        available = [c for c in ANNOTATION_COLS if c in gene_df.columns]
        raise ValueError(f"Target column '{target_col}' not found. Available: {available}")
    else:
        gene_df[target_col] = gene_df[target_col].fillna(0.0)

    return gene_df, feat_cols
