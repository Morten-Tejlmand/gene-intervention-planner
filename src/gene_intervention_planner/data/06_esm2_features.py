"""
Step 6 — ESM-2 Protein Language Model Embeddings
==================================================
Uses facebook/esm2_t6_8M_UR50D (8M params, 320-dim hidden) to generate
per-gene protein embeddings. For each WBGene ID the longest isoform is
selected from the FASTA, truncated to MAX_LEN amino acids, and fed through
ESM-2.  Residue-level embeddings are mean-pooled (excluding padding and
special tokens) to produce a single 320-dim vector per gene.

Results are cached to data/processed/esm2_embeddings.csv so subsequent
runs are instant.

Run from project root:
    python src/gene_intervention_planner/data/06_esm2_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

FASTA_PATH = Path("data/raw/wormbase/caenorhabditis_elegans.PRJNA13758.WBPS19.protein.fa")
OUT_CSV = Path("data/processed/esm2_embeddings.csv")
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
MAX_LEN = 128   # first 128 AA per gene; ~4x faster than 256 with minor quality loss
BATCH_SIZE = 32


def get_representative_sequences() -> dict[str, str]:
    """Return longest isoform sequence (str) per WBGene ID."""
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


def compute_esm2_embeddings(gene_seqs: dict[str, str]) -> pd.DataFrame:
    try:
        import torch
        from transformers import EsmModel, EsmTokenizer
    except ImportError:
        sys.exit("ERROR: install torch and transformers first:\n"
                 "  uv pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                 "  uv pip install transformers")

    print(f"  Loading {MODEL_NAME} ...")
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model.eval()

    gene_ids = list(gene_seqs.keys())
    # Truncate sequences before tokenisation to control runtime
    sequences = [gene_seqs[g][:MAX_LEN] for g in gene_ids]

    all_embeddings: list[np.ndarray] = []
    n_batches = (len(sequences) + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.no_grad():
        for i in range(n_batches):
            if i % 200 == 0:
                pct = 100 * i / n_batches
                print(f"    Batch {i}/{n_batches}  ({pct:.1f}%) ...")
            batch_seqs = sequences[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN + 2,  # +2 for [CLS] / [EOS]
            )
            outputs = model(**inputs)
            # Mean-pool over actual tokens, ignore padding
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            token_emb = outputs.last_hidden_state
            mean_emb = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embeddings.append(mean_emb.numpy())

    embeddings = np.vstack(all_embeddings)  # (n_genes, 320)
    cols = [f"esm2_{i}" for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=cols)
    df.insert(0, "common_name", gene_ids)
    return df


def main() -> None:
    print("=" * 60)
    print("Step 6: ESM-2 Protein Embeddings")
    print("=" * 60)

    if OUT_CSV.exists():
        print(f"  Cached: {OUT_CSV}  —  skipping computation")
        return

    if not FASTA_PATH.exists():
        sys.exit(f"ERROR: FASTA not found: {FASTA_PATH}")

    print(f"  Loading sequences from {FASTA_PATH.name} ...")
    gene_seqs = get_representative_sequences()
    print(f"  Unique WBGene IDs: {len(gene_seqs):,}")
    print(f"  Max AA length used: {MAX_LEN}  |  Batch size: {BATCH_SIZE}")

    df = compute_esm2_embeddings(gene_seqs)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"  Saved {df.shape} -> {OUT_CSV}")


if __name__ == "__main__":
    main()
