"""
Step 8 — Gene Ontology (GO) Term Features
===========================================
Downloads the WormBase C. elegans GO annotation GAF from the Gene Ontology
Consortium (wb.gaf.gz), builds a gene × GO-term multi-hot matrix, then
compresses to N_COMPONENTS dense features via TruncatedSVD.

GO terms span three namespaces:
  BP — Biological Process
  MF — Molecular Function
  CC — Cellular Component

Only positive (non-NOT) annotations are used.  GO terms annotating fewer
than MIN_GENE_COUNT genes are discarded to reduce noise.

Run from project root:
    python src/gene_intervention_planner/data/08_go_features.py
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

GO_GAF_URL = "https://current.geneontology.org/annotations/wb.gaf.gz"
GO_GAF_CACHE = Path("data/raw/wormbase/wb_go_annotations.gaf.gz")
OUT_CSV = Path("data/processed/go_features.csv")
N_COMPONENTS = 64
MIN_GENE_COUNT = 5

_GAF_COLS = [
    "db", "gene_id", "gene_symbol", "qualifier", "go_id",
    "reference", "evidence_code", "with_from", "aspect",
    "db_object_name", "db_object_synonym", "db_object_type",
    "taxon", "date", "assigned_by", "annotation_extension", "gene_product_form",
]


def download_go_annotations() -> None:
    if GO_GAF_CACHE.exists():
        print(f"  GO annotation cache found: {GO_GAF_CACHE}")
        return
    print(f"  Downloading GO annotations from:\n    {GO_GAF_URL}")
    GO_GAF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(GO_GAF_URL, stream=True, timeout=120)
    r.raise_for_status()
    with open(GO_GAF_CACHE, "wb") as fh:
        for chunk in r.iter_content(chunk_size=65536):
            fh.write(chunk)
    print(f"  Downloaded -> {GO_GAF_CACHE}")


def parse_go_gaf(path: Path) -> pd.DataFrame:
    n_cols = len(_GAF_COLS)
    rows: list[list[str]] = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("!"):
                continue
            parts = line.split("\t")
            parts += [""] * (n_cols - len(parts))
            rows.append(parts[:n_cols])
    return pd.DataFrame(rows, columns=_GAF_COLS)


def build_go_features() -> pd.DataFrame:
    print("  Parsing GO annotations ...")
    gaf = parse_go_gaf(GO_GAF_CACHE)

    # Keep only positive annotations on WBGene IDs
    is_not = gaf["qualifier"].str.upper().str.contains("NOT").fillna(False)
    gaf = gaf[~is_not]
    gaf = gaf[gaf["gene_id"].str.startswith("WBGene")]

    print(
        f"  Annotations: {len(gaf):,}  |  "
        f"Genes: {gaf['gene_id'].nunique():,}  |  "
        f"GO terms: {gaf['go_id'].nunique():,}"
    )

    # Filter rare GO terms
    term_gene_counts = gaf.groupby("go_id")["gene_id"].nunique()
    valid_terms = term_gene_counts[term_gene_counts >= MIN_GENE_COUNT].index
    gaf = gaf[gaf["go_id"].isin(valid_terms)]
    gaf = gaf.drop_duplicates(["gene_id", "go_id"])
    print(
        f"  After MIN_GENE_COUNT>={MIN_GENE_COUNT}: "
        f"{gaf['go_id'].nunique():,} GO terms  |  {gaf['gene_id'].nunique():,} genes"
    )

    # Build sparse multi-hot matrix
    gene_ids = sorted(gaf["gene_id"].unique())
    go_terms = sorted(gaf["go_id"].unique())
    gene_idx = {g: i for i, g in enumerate(gene_ids)}
    term_idx = {t: i for i, t in enumerate(go_terms)}

    row_i = [gene_idx[r] for r in gaf["gene_id"]]
    col_j = [term_idx[t] for t in gaf["go_id"]]
    X_sparse = csr_matrix(
        (np.ones(len(row_i), dtype=np.float32), (row_i, col_j)),
        shape=(len(gene_ids), len(go_terms)),
    )
    print(f"  Sparse matrix: {X_sparse.shape}  |  Non-zeros: {X_sparse.nnz:,}")

    # L2-normalize rows then TruncatedSVD
    X_norm = normalize(X_sparse, norm="l2")
    n_components = min(N_COMPONENTS, X_norm.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X_norm)
    var_explained = svd.explained_variance_ratio_.sum()
    print(f"  SVD {n_components} components explain {var_explained:.3f} of variance")

    cols = [f"go_svd_{i}" for i in range(n_components)]
    df = pd.DataFrame(X_svd, columns=cols)
    df.insert(0, "common_name", gene_ids)
    return df


def main() -> None:
    print("=" * 60)
    print("Step 8: GO Term Features (SVD-compressed)")
    print("=" * 60)

    if OUT_CSV.exists():
        print(f"  Cached: {OUT_CSV}  —  skipping")
        return

    download_go_annotations()
    df = build_go_features()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"  Saved {df.shape} -> {OUT_CSV}")


if __name__ == "__main__":
    main()
