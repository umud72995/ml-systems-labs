# Data Engineering Task Formulation: Census Income

**Developer:** Umud Kalbizada (ID: 72995)  
**Cohort:** Adult Census Dataset (OpenML, v2)  

### 1. Problem Formulation
Construct a vectorized, leakage-free data pipeline predicting whether individual income exceeds $50K using survey records from the 1994 US census.

### 2. Feature Schema & Evaluation
- Inputs: 14 demographic and employment predictors.
- Target: Binary income label (`>50K`).
- Evaluation: **F1-Score** is prioritized to handle class imbalance (~76/24 split).
