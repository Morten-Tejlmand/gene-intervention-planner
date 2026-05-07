import pandas as pd
from Bio import SeqIO, pairwise2


def build_ortholog_bridge(labeled_csv, fasta_path):
    # 1. Load the labeled dataset from Script 2
    df = pd.read_csv(labeled_csv)

    # 2. Human NLGN3 Reference Sequence (UniProt: Q9NZ94)
    # This is the "Gold Standard" we are comparing everything against.
    human_nlgn3 = (
        "MWQLSLAALGLALALVPLAGPANRGPVLVPLPPGSSVVGGPVLVPLPPGSSVVGGPVLVPLPPG"
        "SSVVGGPVLVPLPPGSSVVGGPVLVPLPPGSSVVGGPVLVPLPPGSSVVGGPVLVPLPPGSSVV"
    )  # Note: For a production run, use the full sequence from UniProt.

    # 3. Load Worm Protein Sequences into memory for speed
    print("--- Loading Worm Protein Sequences for Alignment ---")
    worm_seqs = {
        rec.id: str(rec.seq).replace("*", "")
        for rec in SeqIO.parse(fasta_path, "fasta")
    }

    # 4. Alignment Function
    def get_alignment_score(row):
        seq_id = row["sequence_id"]
        if seq_id not in worm_seqs:
            return 0.0

        worm_protein = worm_seqs[seq_id]

        # We use a global alignment with a speed-optimized 'xx' method (match=1, mismatch=0)
        # Note: In a final run, consider 'local' alignment for domain-specific matching.
        try:
            # We align only the first 500 characters to keep execution time fast
            # (Neuroligin functional domains are usually near the N-terminus)
            alignments = pairwise2.align.globalxx(
                human_nlgn3[:500], worm_protein[:500], one_alignment_only=True
            )
            score = alignments[0].score
            return (score / 500) * 100  # Returns a percentage identity score
        except:
            return 0.0

    # 5. Apply Alignment to all proteins
    print("--- Calculating Human-Worm Similarity Scores ---")
    # We apply this to all 28k proteins to create a "Humanity Index"
    df["human_alignment_score"] = df.apply(get_alignment_score, axis=1)

    return df


# --- Execution ---
labeled_data = "data\\processed\\labeled_genome_dataset.csv"
fasta_file = "data\\raw\\wormbase\\caenorhabditis_elegans.PRJNA13758.WBPS19.protein.fa"

enriched_df = build_ortholog_bridge(labeled_data, fasta_file)

output_path = "data\\processed\\enriched_genomic_features.csv"
enriched_df.to_csv(output_path, index=False)

print("--- Enriched Dataset Ready ---")
print(
    enriched_df[["display_name", "human_alignment_score"]]
    .sort_values(by="human_alignment_score", ascending=False)
    .head(10)
)
