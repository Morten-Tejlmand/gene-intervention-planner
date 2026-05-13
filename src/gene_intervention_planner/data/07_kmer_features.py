"""
Step 7 — Dipeptide Composition Features
=========================================
Computes 400-dimensional dipeptide (k=2) frequency profiles for every gene.
Dipeptide composition captures the conditional frequency of adjacent amino
acid pairs — a compact, alignment-free sequence encoding that complements
biophysical aggregate statistics.

For each WBGene ID the longest isoform is used.

Run from project root:
    python src/gene_intervention_planner/data/07_kmer_features.py
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

FASTA_PATH = Path("data/raw/wormbase/caenorhabditis_elegans.PRJNA13758.WBPS19.protein.fa")
OUT_CSV = Path("data/processed/kmer_features.csv")

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
DIPEPTIDES = ["".join(p) for p in product(AMINO_ACIDS, repeat=2)]  # 400 features


def get_representative_sequences() -> dict[str, str]:
    gene_seqs: dict[str, str] = {}
    for rec in SeqIO.parse(str(FASTA_PATH), "fasta"):
        gene_id = None
        for part in rec.description.split():
            if part.startswith("gene="):
                gene_id = part.split("=")[1]
                break
        if gene_id is None:
            continue
        seq = str(rec.seq)
        if gene_id not in gene_seqs or len(seq) > len(gene_seqs[gene_id]):
            gene_seqs[gene_id] = seq
    return gene_seqs


def dipeptide_composition(seq: str) -> np.ndarray:
    """400-dim normalized dipeptide frequency vector."""
    seq = seq.upper()
    valid = set(AMINO_ACIDS)
    total = len(seq) - 1
    if total <= 0:
        return np.zeros(400)
    dp_index = {dp: i for i, dp in enumerate(DIPEPTIDES)}
    counts = np.zeros(400)
    for i in range(len(seq) - 1):
        dp = seq[i: i + 2]
        if dp in dp_index:
            counts[dp_index[dp]] += 1
    return counts / max(total, 1)


def main() -> None:
    print("=" * 60)
    print("Step 7: Dipeptide Composition Features")
    print("=" * 60)

    if OUT_CSV.exists():
        print(f"  Cached: {OUT_CSV}  —  skipping")
        return

    if not FASTA_PATH.exists():
        sys.exit(f"ERROR: FASTA not found: {FASTA_PATH}")

    print(f"  Loading sequences ...")
    gene_seqs = get_representative_sequences()
    print(f"  Unique WBGene IDs: {len(gene_seqs):,}")

    rows: list[list] = []
    n = len(gene_seqs)
    for i, (gene_id, seq) in enumerate(gene_seqs.items()):
        if i % 5000 == 0:
            print(f"    {i}/{n} ...")
        dc = dipeptide_composition(seq)
        rows.append([gene_id] + dc.tolist())

    cols = ["common_name"] + DIPEPTIDES
    df = pd.DataFrame(rows, columns=cols)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"  Saved {df.shape} -> {OUT_CSV}")


if __name__ == "__main__":
    main()
