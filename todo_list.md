


** Right now beraly better than random in most experiments**
proposed method for fixing this, the know genes is low, so it has a very small start pool to draft from
- More seed genes, expand NEURAL_GENE_IDS to 100+ verified neural WBGene IDs
- Phenotype-based labeling: parse data/raw/wormbase/*.phenotypes.gaf and label any gene with locomotion/synaptic phenotype annotations as positive
- Restrict the pool: work only within annotated genes instead of the full proteome
- split genes into sub genes? 