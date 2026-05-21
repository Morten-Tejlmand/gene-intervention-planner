import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    "src/gene_intervention_planner/data/01_dataset_all.py",
    "src/gene_intervention_planner/data/02_label_genes.py",
    "src/gene_intervention_planner/data/03_ortholog_mapper.py",
    "src/gene_intervention_planner/data/04_synptic_mapping.py",
    "src/gene_intervention_planner/data/05_functional_features.py",
    "src/gene_intervention_planner/data/06_esm2_features.py",
    "src/gene_intervention_planner/data/07_kmer_features.py",
    "src/gene_intervention_planner/data/08_go_features.py",
]

for script in STEPS:
    name = Path(script).name
    print(f"\n>>> {name}")
    t0 = time.time()
    ret = subprocess.run([sys.executable, script])
    elapsed = time.time() - t0
    if ret.returncode != 0:
        print("    FAILED — stopping.")
        sys.exit(1)
    print(f"    done in {elapsed:.0f}s")

print("\nAll data steps complete.")
