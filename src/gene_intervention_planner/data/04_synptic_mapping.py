import pandas as pd


def build_ppi_matrix(features_csv):
    # 1. Load the enriched data (from Script 3)
    df = pd.read_csv(features_csv)

    # 2. Identify the "Locks" and "Keys"
    # We'll isolate all nlg-1 variants and all nrx-1 variants
    nlg_variants = df[df["display_name"] == "nlg-1"].copy()
    nrx_variants = df[df["display_name"] == "nrx-1"].copy()

    # If no nlg-1/nrx-1 are found, we use the top candidates the AI is suspicious of
    if nlg_variants.empty or nrx_variants.empty:
        print(
            "Warning: nlg-1 or nrx-1 not found in display_name. Using high-risk candidates."
        )
        nlg_variants = df.nlargest(5, "human_alignment_score")
        nrx_variants = df.nsmallest(5, "isoelectric_point")

    interaction_results = []

    print(
        f"--- Modeling {len(nlg_variants)} x {len(nrx_variants)} Interaction Grid ---"
    )

    # 3. Calculate "Fit" scores for every possible pair
    for _, nlg in nlg_variants.iterrows():
        for _, nrx in nrx_variants.iterrows():
            # Feature A: Hydrophobic Complementarity
            # Similarity in GRAVY scores suggests they can exist in the same membrane environment
            hydro_fit = 1 / (1 + abs(nlg["gravy"] - nrx["gravy"]))

            # Feature B: Electrostatic Attraction
            # Significant differences in pI (Isoelectric Point) suggest opposite charges
            # which facilitates "stickiness" at synaptic pH.
            charge_diff = abs(nlg["isoelectric_point"] - nrx["isoelectric_point"])

            # Feature C: Stability Product
            # If BOTH are unstable (high instability_index), the interaction fails.
            joint_stability = 1 / (
                1 + (nlg["instability_index"] * nrx["instability_index"] / 1000)
            )

            # Composite Interaction Score (The "Synergy" Score)
            synergy_score = (
                (hydro_fit * 0.3) + (charge_diff * 0.4) + (joint_stability * 0.3)
            )

            interaction_results.append(
                {
                    "nlg_isoform": nlg["sequence_id"],
                    "nrx_isoform": nrx["sequence_id"],
                    "interaction_strength": synergy_score,
                    "is_stable_pair": 1 if synergy_score > 0.75 else 0,
                }
            )

    # 4. Create the Interaction Matrix
    ppi_df = pd.DataFrame(interaction_results)
    return ppi_df


# --- Execution ---
enriched_data = "data\\processed\\enriched_genomic_features.csv"
ppi_matrix = build_ppi_matrix(enriched_data)

# Save the PPI Matrix
output_path = "data\\processed\\nlg_nrx_interaction_matrix.csv"
ppi_matrix.to_csv(output_path, index=False)

print("\n--- PPI Analysis Complete ---")
print(f"Total potential synaptic pairs modeled: {len(ppi_matrix)}")
print(f"Top interaction strength found: {ppi_matrix['interaction_strength'].max():.4f}")
print(ppi_matrix.sort_values(by="interaction_strength", ascending=False).head())
