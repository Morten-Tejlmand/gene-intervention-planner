import re

import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def extract_universal_features(fasta_path):
    protein_data = []
    print(f"--- Processing: {fasta_path} ---")

    for record in SeqIO.parse(fasta_path, "fasta"):
        # 1. NEW: Extract Common Name from the FASTA Header
        # Usually looks like: ID=2L52.1a gene=nlg-1 locus=...
        common_name = record.id  # Default to ID if no name found
        name_match = re.search(r"gene=([\w-]+)", record.description)
        if name_match:
            common_name = name_match.group(1)

        raw_seq = str(record.seq).upper().replace("*", "")
        clean_seq = raw_seq.replace("U", "C").replace("X", "").replace("O", "")

        if len(clean_seq) < 20:
            continue

        try:
            analysed_seq = ProteinAnalysis(clean_seq)

            features = {
                "sequence_id": record.id,
                "common_name": common_name,
                "length": len(clean_seq),
                "molecular_weight": analysed_seq.molecular_weight(),
                "aromaticity": analysed_seq.aromaticity(),
                "instability_index": analysed_seq.instability_index(),
                "isoelectric_point": analysed_seq.isoelectric_point(),
                "gravy": analysed_seq.gravy(),
            }

            # Secondary Structure
            sec_struc = analysed_seq.secondary_structure_fraction()
            (
                features["helix_fraction"],
                features["turn_fraction"],
                features["sheet_fraction"],
            ) = sec_struc

            # Amino Acid Percentages
            aa_percentages = analysed_seq.get_amino_acids_percent()
            for aa, percent in aa_percentages.items():
                features[f"percent_{aa}"] = percent

            protein_data.append(features)

        except Exception:
            continue

    df = pd.DataFrame(protein_data)
    return df


# --- Execution ---
# Note: Using your specific path from the snippet
fasta_file = "data\\raw\\wormbase\\caenorhabditis_elegans.PRJNA13758.WBPS19.protein.fa"
universal_matrix = extract_universal_features(fasta_file)

# Save to your processed folder
output_path = "data\\processed\\universal_protein_features.csv"
universal_matrix.to_csv(output_path, index=False)

print(f"Success! Extracted {len(universal_matrix)} protein profiles.")
# Check if common_name is populated
print(universal_matrix[["sequence_id", "common_name"]].head())
