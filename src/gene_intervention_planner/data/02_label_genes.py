import pandas as pd


def build_labeled_dataset(gff_path, features_csv):
    # 1. Load the features from Script 1
    df = pd.read_csv(features_csv)

    # 2. Define our "Mental Risk" Seed IDs directly
    # These correspond to: nlg-1, nrx-1, unc-13, ric-4, snb-1
    # (Verified for C. elegans PRJNA13758)
    seed_wbids = {
        "WBGene00003734": "nlg-1",
        "WBGene00003816": "nrx-1",
        "WBGene00006745": "unc-13",
        "WBGene00004354": "ric-4",
        "WBGene00004944": "snb-1",
    }

    print("--- Seeding Genome using WBIDs ---")

    # 3. Apply the Label
    # 'common_name' in your CSV currently holds values like 'WBGene00007063'
    # We check if that ID is in our seed list
    df["is_mental_risk"] = df["common_name"].apply(
        lambda x: 1 if str(x) in seed_wbids.keys() else 0
    )

    # 4. Create a clean Display Name for humans
    # If the ID is in our seed list, use the name (e.g., nlg-1), otherwise keep the ID
    df["display_name"] = df["common_name"].map(seed_wbids).fillna(df["common_name"])

    return df


# --- Execution ---
processed_features = "data\\processed\\universal_protein_features.csv"
labeled_df = build_labeled_dataset(
    None, processed_features
)  # GFF no longer strictly needed for seeding

# Save the final version
output_path = "data\\processed\\labeled_genome_dataset.csv"
labeled_df.to_csv(output_path, index=False)

print("--- SUCCESS ---")
print(f"Total Seeded Targets (1s): {labeled_df['is_mental_risk'].sum()}")
print(labeled_df[labeled_df["is_mental_risk"] == 1][["sequence_id", "display_name"]])
