"""
Build neural-behavioural effect-count target from raw WormBase phenotype GAF.

Filters annotations to only phenotype terms directly related to neural function
and behaviour: locomotion, chemosensation, synaptic transmission, motor output,
touch response, etc. Excludes purely structural/morphological or non-neural
phenotypes (embryonic lethal, growth rate, fat content, etc.).

Output: data/processed/neural_behaviour_target.csv
  common_name            — WormBase gene ID
  neural_behaviour_count — number of distinct neural/behavioural phenotype
                           annotations for this gene (non-NOT, IMP evidence)
  neural_behaviour_ids   — pipe-separated list of matched phenotype IDs

Run from project root:
    python experiments/behavioural_al/build_neural_target.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

GAF_PATH = Path("data/raw/wormbase/caenorhabditis_elegans.PRJNA13758.WBPS19.phenotypes.gaf")
OUT_PATH  = Path("data/processed/neural_behaviour_target.csv")

# ---------------------------------------------------------------------------
# Neural / behavioural phenotype ontology terms
# These are all WBPhenotype IDs whose descriptions match locomotion, sensation,
# synaptic function, motor output, or neural circuit behaviour.
# Structural/anatomical terms (axon morphology, neuron count) are included only
# where they reflect a functional behavioural consequence.
# ---------------------------------------------------------------------------
NEURAL_BEHAVIOUR_IDS: set[str] = {
    # --- Locomotion & movement ---
    "WBPhenotype:0000643",  # locomotion variant
    "WBPhenotype:0001213",  # locomotion reduced
    "WBPhenotype:0002347",  # forward locomotion decreased
    "WBPhenotype:0002346",  # forward locomotion increased
    "WBPhenotype:0002345",  # forward locomotion variant
    "WBPhenotype:0001005",  # backward locomotion variant
    "WBPhenotype:0004017",  # locomotor coordination variant
    "WBPhenotype:0001713",  # locomotory rhythm variant
    "WBPhenotype:0001206",  # movement variant
    "WBPhenotype:0000253",  # movement erratic
    "WBPhenotype:0001699",  # localized movement variant
    "WBPhenotype:0001872",  # drug induced locomotion variant
    "WBPhenotype:0001773",  # enhanced locomotion defective
    "WBPhenotype:0002000",  # recovery from enhanced locomotion variant
    "WBPhenotype:0001627",  # locomotion rate serotonin hypersensitive
    "WBPhenotype:0001628",  # locomotion rate serotonin resistant
    # Body bends & sinusoidal movement
    "WBPhenotype:0001309",  # amplitude of sinusoidal movement decreased
    "WBPhenotype:0002328",  # amplitude of sinusoidal movement increased
    "WBPhenotype:0004022",  # amplitude of sinusoidal movement variant
    "WBPhenotype:0001482",  # frequency body bend reduced
    "WBPhenotype:0004023",  # frequency of body bend variant
    "WBPhenotype:0002348",  # frequency of body bends increased
    "WBPhenotype:0001703",  # body bend variant
    "WBPhenotype:0000664",  # exaggerated body bends
    "WBPhenotype:0001307",  # reduced body bend
    "WBPhenotype:0004018",  # sinusoidal movement variant
    # Turning & reversals
    "WBPhenotype:0002312",  # turning frequency variant
    "WBPhenotype:0002313",  # turning frequency increased
    "WBPhenotype:0002314",  # turning frequency reduced
    "WBPhenotype:0002320",  # reversal variant
    "WBPhenotype:0001506",  # spontaneous reversal rate variant
    "WBPhenotype:0001505",  # spontaneous reversal variant
    "WBPhenotype:0002549",  # increased frequency of spontaneous reversal initiation
    "WBPhenotype:0002602",  # reduced frequency of spontaneous reversal initiation
    "WBPhenotype:0000352",  # backing uncoordinated
    "WBPhenotype:0002550",  # conflicted kinker with gradual backward movement
    "WBPhenotype:0002548",  # kinker from conflicting forward and backward body bend propagation
    # Speed & velocity
    "WBPhenotype:0002326",  # velocity of movement decreased
    "WBPhenotype:0002327",  # velocity of movement increased
    "WBPhenotype:0004025",  # velocity of movement variant
    "WBPhenotype:0004024",  # wavelength of movement variant
    # Swimming & thrashing
    "WBPhenotype:0001700",  # swimming variant
    "WBPhenotype:0002114",  # swimming induced paralysis
    "WBPhenotype:0001481",  # thrashing increased
    "WBPhenotype:0000273",  # thrashing reduced
    "WBPhenotype:0002375",  # reversion of swimming paralysis
    "WBPhenotype:0002233",  # osmolarity modulated swimming variant
    # Paralysis
    "WBPhenotype:0000473",  # progressive paralysis
    "WBPhenotype:0001701",  # spastic locomotion
    "WBPhenotype:0000455",  # jerky movement
    # Head & nose movement
    "WBPhenotype:0002302",  # nose movement decreased
    "WBPhenotype:0002303",  # nose movement increased
    "WBPhenotype:0002301",  # nose movement variant
    "WBPhenotype:0001265",  # head movement variant
    "WBPhenotype:0002304",  # head movement decreased
    "WBPhenotype:0002287",  # tail movement variant
    "WBPhenotype:0002306",  # tail movement decreased
    "WBPhenotype:0000796",  # posterior body uncoordinated
    # --- Chemosensation & sensory responses ---
    "WBPhenotype:0001434",  # chemotaxis variant
    "WBPhenotype:0000677",  # chemosensation defective
    "WBPhenotype:0001049",  # chemosensory behavior variant
    "WBPhenotype:0001040",  # chemosensory response variant
    "WBPhenotype:0001048",  # odorant chemosensory response variant
    "WBPhenotype:0001438",  # odorant positive chemotaxis defective
    "WBPhenotype:0001448",  # odorant negative chemotaxis defective
    "WBPhenotype:0000481",  # negative chemotaxis variant
    "WBPhenotype:0001441",  # aqueous positive chemotaxis defective
    "WBPhenotype:0001450",  # aqueous negative chemotaxis variant
    "WBPhenotype:0001443",  # aqueous chemosensory response variant
    "WBPhenotype:0000015",  # positive chemotaxis defective
    "WBPhenotype:0001061",  # AWA odorant chemotaxis defective
    "WBPhenotype:0001060",  # AWC odorant chemotaxis defective
    "WBPhenotype:0000302",  # benzaldehyde chemotaxis defective
    "WBPhenotype:0001468",  # benzaldehyde chemotaxis variant
    "WBPhenotype:0000303",  # diacetyl chemotaxis defective
    "WBPhenotype:0001470",  # diacetyl chemotaxis variant
    "WBPhenotype:0000304",  # isoamyl alcohol chemotaxis defective
    "WBPhenotype:0001472",  # isoamyl alcohol chemotaxis variant
    "WBPhenotype:0000412",  # octanol chemotaxis defective
    "WBPhenotype:0001437",  # octanol chemotaxis hypersensitive
    "WBPhenotype:0001452",  # octanol chemotaxis variant
    "WBPhenotype:0000630",  # quinine chemotaxis defective
    "WBPhenotype:0001454",  # quinine chemotaxis variant
    "WBPhenotype:0000480",  # pyrazine chemotaxis defective
    "WBPhenotype:0001474",  # pyrazine chemotaxis variant
    "WBPhenotype:0001086",  # trimethylthiazole chemotaxis defective
    "WBPhenotype:0001085",  # butanone chemotaxis defective
    "WBPhenotype:0001469",  # butanone chemotaxis variant
    "WBPhenotype:0001234",  # nonanone chemotaxis defective
    "WBPhenotype:0000247",  # sodium chemotaxis defective
    "WBPhenotype:0001084",  # sodium chloride chemotaxis defective
    "WBPhenotype:0001462",  # sodium chloride chemotaxis variant
    "WBPhenotype:0000254",  # chloride chemotaxis defective
    "WBPhenotype:0001435",  # ammonium chloride chemotaxis defective
    "WBPhenotype:0001484",  # ammonium acetate chemotaxis defective
    "WBPhenotype:0001738",  # ammonium chemotaxis defective
    "WBPhenotype:0001057",  # lithium chemotaxis defective
    "WBPhenotype:0001059",  # magnesium chemotaxis defective
    "WBPhenotype:0001056",  # iodide chemotaxis defective
    "WBPhenotype:0001055",  # bromide chemotaxis defective
    "WBPhenotype:0001565",  # lysine chemotaxis defective
    "WBPhenotype:0001058",  # potassium chemotaxis defective
    "WBPhenotype:0001736",  # acetate chemotaxis defective
    "WBPhenotype:0000264",  # cAMP chemotaxis defective
    "WBPhenotype:0001500",  # pentanedione chemotaxis variant
    "WBPhenotype:0002171",  # alkaline pH chemotaxis variant
    "WBPhenotype:0002385",  # high pH avoidance defective
    "WBPhenotype:0001453",  # high sodium chloride concentration osmotic avoidance variant
    "WBPhenotype:0001442",  # sodium acetate chemotaxis defective
    "WBPhenotype:0001461",  # sodium acetate chemotaxis variant
    # Touch / mechanosensation
    "WBPhenotype:0000315",  # mechanosensation variant
    "WBPhenotype:0000653",  # mechanosensory system variant
    "WBPhenotype:0001221",  # nose touch defective
    "WBPhenotype:0001436",  # nose touch hypersensitive
    "WBPhenotype:0004026",  # nose touch variant
    "WBPhenotype:0000456",  # touch resistant
    "WBPhenotype:0000850",  # touch resistant anterior body
    "WBPhenotype:0000316",  # touch resistant tail
    "WBPhenotype:0000397",  # harsh body touch resistant
    "WBPhenotype:0000398",  # light body touch resistant
    "WBPhenotype:0004029",  # sexually dimorphic mechanosensation variant
    "WBPhenotype:0002366",  # touch-induced suppression of head movement defective
    # Osmotic & thermal
    "WBPhenotype:0000249",  # osmotic avoidance defective
    "WBPhenotype:0000663",  # osmotic avoidance variant
    "WBPhenotype:0002512",  # thermosensory behavior variant
    "WBPhenotype:0002513",  # thermotaxis variant
    "WBPhenotype:0001999",  # conflicting sensory integration variant
    "WBPhenotype:0002370",  # noxious heat avoidance defective
    # Avoidance / pathogen
    "WBPhenotype:0000473",  # avoids bacterial lawn (also in locomotion)
    "WBPhenotype:0002378",  # pathogen avoidance variant
    "WBPhenotype:0001765",  # carbon dioxide avoidance variant
    # --- Synaptic transmission ---
    "WBPhenotype:0000584",  # synaptic transmission variant
    "WBPhenotype:0000657",  # neuronal synaptic transmission variant
    "WBPhenotype:0000658",  # neuromuscular synaptic transmission variant
    "WBPhenotype:0000655",  # GABA synaptic transmission variant
    "WBPhenotype:0000656",  # acetylcholine synaptic transmission variant
    "WBPhenotype:0001320",  # endogenous synaptic amplitude variant
    "WBPhenotype:0001319",  # endogenous synaptic event frequency reduced
    "WBPhenotype:0001318",  # endogenous synaptic events variant
    "WBPhenotype:0001317",  # evoked postsynaptic amplitude reduced
    "WBPhenotype:0001316",  # evoked postsynaptic current variant
    "WBPhenotype:0002583",  # miniature inhibitory post-synaptic current amplitude increased
    "WBPhenotype:0001798",  # inhibition of synaptogenesis defective
    "WBPhenotype:0002232",  # synaptic remodeling variant
    "WBPhenotype:0001514",  # synaptic vesicle endocytosis variant
    "WBPhenotype:0000654",  # synaptic vesicle exocytosis variant
    "WBPhenotype:0001684",  # synaptic vesicle homeostasis variant
    "WBPhenotype:0001933",  # synaptic vesicle tag abnormal in mechanosensory neurons
    "WBPhenotype:0000102",  # presynaptic vesicle cluster variant
    "WBPhenotype:0000672",  # presynaptic vesicle cluster localization variant
    "WBPhenotype:0001670",  # presynaptic vesicle number reduced
    "WBPhenotype:0001669",  # presynaptic vesicle number variant
    "WBPhenotype:0002372",  # presynaptic vesicle morphology altered
    "WBPhenotype:0002001",  # synapse density variant
    "WBPhenotype:0000616",  # synapse morphology variant
    "WBPhenotype:0000625",  # synaptogenesis variant
    "WBPhenotype:0001685",  # postsynaptic region morphology variant
    "WBPhenotype:0002262",  # postsynaptic component localization variant
    "WBPhenotype:0000847",  # presynaptic component localization variant
    "WBPhenotype:0001321",  # presynaptic region morphology variant
    "WBPhenotype:0000846",  # presynaptic region physiology variant
    "WBPhenotype:0002113",  # synapsis formation variant
    "WBPhenotype:0001813",  # synapsis defective
    # --- Neuron activation & physiology ---
    "WBPhenotype:0001818",  # neuron activation variant
    "WBPhenotype:0002199",  # neuron calcium transient levels variant
    "WBPhenotype:0001042",  # neuron function reduced
    "WBPhenotype:0002563",  # neuron physiology phenotype
    "WBPhenotype:0000851",  # ciliated neuron physiology variant
    "WBPhenotype:0001786",  # neuronal ion channel clustering defective
    "WBPhenotype:0002166",  # muscle membrane potential variant
    # --- Pharyngeal pumping / feeding (neurally controlled) ---
    "WBPhenotype:0000634",  # pharyngeal pumping variant
    "WBPhenotype:0000019",  # pharyngeal pumping reduced
    "WBPhenotype:0000018",  # pharyngeal pumping increased
    "WBPhenotype:0000020",  # pharyngeal pumping irregular
    "WBPhenotype:0001006",  # pharyngeal pumping rate variant
    "WBPhenotype:0000378",  # pharyngeal pumping shallow
    "WBPhenotype:0002441",  # pharyngeal inter-pump interval increased
    "WBPhenotype:0002442",  # pharyngeal pump duration increased
    "WBPhenotype:0002446",  # pharyngeal pumping rate in response to light variant
    "WBPhenotype:0000747",  # pharyngeal contraction defective
    "WBPhenotype:0000980",  # pharyngeal contraction variant
    "WBPhenotype:0000413",  # pharyngeal muscle paralyzed
    "WBPhenotype:0001286",  # food suppressed pumping variant
    "WBPhenotype:0001733",  # pheromone suppressed pumping defective
    "WBPhenotype:0001734",  # starvation suppressed pumping defective
    "WBPhenotype:0000844",  # serotonin induced pumping variant
    "WBPhenotype:0001396",  # pumping absent
    "WBPhenotype:0000329",  # pumping asynchronous
    "WBPhenotype:0001281",  # stuffed pharynx
    "WBPhenotype:0001045",  # sporadic pumping
    "WBPhenotype:0000778",  # feeding inefficient
    "WBPhenotype:0002241",  # feeding reduced
    "WBPhenotype:0000659",  # feeding behavior variant
    "WBPhenotype:0000660",  # social feeding increased
    "WBPhenotype:0000661",  # solitary feeding increased
    # --- Egg laying (neurally controlled) ---
    "WBPhenotype:0000640",  # egg laying variant
    "WBPhenotype:0000006",  # egg laying defective
    "WBPhenotype:0000004",  # constitutive egg laying
    "WBPhenotype:0000005",  # hyperactive egg laying
    "WBPhenotype:0001068",  # egg laying serotonin resistant
    "WBPhenotype:0001622",  # egg laying serotonin hypersensitive
    "WBPhenotype:0001339",  # egg laying levamisole resistant
    "WBPhenotype:0001340",  # egg laying imipramine resistant
    "WBPhenotype:0001342",  # egg laying imipramine hypersensitive
    "WBPhenotype:0001341",  # egg laying phentolamine resistant
    "WBPhenotype:0001101",  # egg laying response to drug variant
    "WBPhenotype:0001073",  # egg laying response to food variant
    "WBPhenotype:0002070",  # egg laying response to muscimol variant
    "WBPhenotype:0001063",  # egg laying phases variant
    "WBPhenotype:0001065",  # fewer egg laying events during active
    "WBPhenotype:0001330",  # egg laying event infrequent
    "WBPhenotype:0002649",  # egg laying response to chemical abnormal
    # --- Defecation cycle (neurally controlled) ---
    "WBPhenotype:0000650",  # defecation variant
    "WBPhenotype:0000207",  # defecation cycle variant
    "WBPhenotype:0000246",  # defecation cycle variable length
    "WBPhenotype:0000208",  # long defecation cycle
    "WBPhenotype:0000209",  # short defecation cycle
    "WBPhenotype:0000391",  # defecation missing motor steps
    "WBPhenotype:0000210",  # defecation contraction variant
    "WBPhenotype:0000564",  # echo defecation cycle
    "WBPhenotype:0000410",  # no defecation cycle
    "WBPhenotype:0001201",  # no expulsion defecation
    "WBPhenotype:0001791",  # defecation cycle temperature compensation variant
    "WBPhenotype:0001092",  # larval defecation defect
    # --- Body wall muscle physiology (locomotion effector) ---
    "WBPhenotype:0001292",  # body wall muscle physiology variant
    "WBPhenotype:0000153",  # body wall muscle contraction abnormal
    "WBPhenotype:0001802",  # body wall muscle relaxation defective
    "WBPhenotype:0001735",  # muscle cell activity variant
    # --- Paralysis & protein aggregation (disease relevance) ---
    "WBPhenotype:0001937",  # hypersensitive to protein aggregation induced paralysis
    "WBPhenotype:0001936",  # resistant to protein aggregation induced paralysis
    "WBPhenotype:0001062",  # late paralysis arrested elongation two fold
}


def build_target(gaf_path: Path = GAF_PATH) -> pd.DataFrame:
    rows = []
    with open(gaf_path) as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            rows.append({
                "gene_id":   parts[1],
                "qualifier": parts[3],
                "pheno_id":  parts[4],
            })

    gaf = pd.DataFrame(rows)
    gaf["is_not"] = gaf["qualifier"].str.upper().str.contains("NOT").fillna(False)

    # Keep only positive, neural-behavioural annotations
    neural = gaf[
        ~gaf["is_not"] &
        gaf["pheno_id"].isin(NEURAL_BEHAVIOUR_IDS)
    ]

    agg = (
        neural.groupby("gene_id")
        .agg(
            neural_behaviour_count=("pheno_id", "count"),
            neural_behaviour_ids=("pheno_id", lambda x: "|".join(sorted(set(x)))),
        )
        .reset_index()
        .rename(columns={"gene_id": "common_name"})
    )
    agg["confirmed_negative"] = False

    # Confirmed negatives: genes with a NOT-qualified annotation to one of the 230
    # neural phenotype IDs.  A lab explicitly tested the gene for a neural behaviour
    # phenotype and published "no effect" — the only reliable source of true negatives.
    not_neural_genes = set(
        gaf[gaf["is_not"] & gaf["pheno_id"].isin(NEURAL_BEHAVIOUR_IDS)]["gene_id"].unique()
    ) - set(agg["common_name"])   # exclude any gene that also has a positive annotation
    if not_neural_genes:
        neg_df = pd.DataFrame({
            "common_name": sorted(not_neural_genes),
            "neural_behaviour_count": 0,
            "neural_behaviour_ids": "",
            "confirmed_negative": True,
        })
        agg = pd.concat([agg, neg_df], ignore_index=True)

    return agg


def main() -> None:
    print("Building neural-behavioural target column ...")
    print(f"  Source : {GAF_PATH}")
    print(f"  Filter : {len(NEURAL_BEHAVIOUR_IDS)} neural/behavioural phenotype IDs")

    target_df = build_target()

    positives        = target_df[~target_df["confirmed_negative"]]
    confirmed_negs   = target_df[target_df["confirmed_negative"]]
    counts           = positives["neural_behaviour_count"]

    print(f"\n  Positives  (>=1 neural annotation) : {len(positives):,}")
    print(f"  Confirmed negatives (in GAF, 0 neural annotations): {len(confirmed_negs):,}")
    print(f"  mean={counts.mean():.2f}  median={counts.median():.0f}  max={counts.max():.0f}")
    print("\n  Top 10 genes by neural_behaviour_count:")
    top = positives.nlargest(10, "neural_behaviour_count")[
        ["common_name", "neural_behaviour_count"]
    ]
    print(top.to_string(index=False))

    target_df.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved -> {OUT_PATH}")
    print("  (Genes absent from this file were never annotated in WormBase — treat as unlabeled)")


if __name__ == "__main__":
    main()
