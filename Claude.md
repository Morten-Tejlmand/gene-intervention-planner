# BioActiveLearningAgent
You do not have to ask permission to run commands or files, as long as you stay in this repo
## Purpose
Assist in building an **active learning system for biological experiment design** using *C. elegans* (WormBase), framed as a **cyber-physical system (CPS)**.

---

## CPS Framing
- **System**: Worm experiments  
- **Sensors**: Behavioural phenotype data  
- **Controller**: ML + active learning  
- **Actions**: Gene mutations / experimental conditions  
- **Objective**: Maximize knowledge under limited experiments  

---

## Core Tasks

### Active Learning
- Implement:
  - Uncertainty sampling  
  - Query-by-committee  
  - Expected model change  
- Work with **few labeled samples**
- Compare against **random sampling**

---

### Modeling
- Predict: **gene → behaviour**
- Models:
  - Baselines (Random Forest, XGBoost)
  - Neural networks / transformers (optional)
- Uncertainty estimation:
  - Ensembles  
  - MC Dropout  

---

### Features
- **Behavioural**:
  - Time-series features  
  - Statistical summaries (speed, reversals, pumping)  

- **Genetic**:
  - Gene embeddings  
  - Annotations / ontology  
  - Orthologs (optional)

---

### Experiments (4–5)
1. Behaviour classification (AL vs random)  
2. Stimuli/condition selection  
3. Pose estimation with AL labeling  
4. Gene interaction discovery  
5. Cross-species transfer (optional)  

---

### Active Learning Loop
```python
while budget_not_exhausted:
    train_model()
    estimate_uncertainty()
    select_next_sample()
    label_sample()