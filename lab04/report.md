# Lab 04: Data Pipeline and Quality Engineering

**Student Researcher:** Umud Kalbizada (Student ID: 72995)  
**Readings:** Reddi, Chapters 5 and 6  

---

## 1. Goal
This laboratory module constructs an automated, production-grade tabular machine learning pipeline for the Adult Census income dataset (`openml.org`, version 2). The pipeline implements lossless raw ingestion logging, pre-flight data schema and range validation under a zero-silent-drop policy, strictly leakage-free train/test partition isolation, feature engineering synthesis benchmarked via Logistic Regression, label noise auditing (conflicting duplicate records), and cryptographic SHA-256 provenance tracking.

---

## 2. Method
1. **Lossless Ingestion:** Ingested the raw dataset (48,842 instances, 15 attributes), recording raw shapes and cryptographic hashes with 0 unlogged row drops.
2. **Schema & Range Contract Verification:** Validated data types, boundary ranges (age in [17, 90], hours-per-week in [1, 99]), and mapped '?' tokens to formal null representations without row deletion prior to partitioning.
3. **Zero-Leakage Transformation:**
   - Applied stratified 70/30 split (`random_state=72995`) based on target income.
   - Fitted median/mode imputers, 1st/99th percentile outlier clipping boundaries, and scalers strictly on the 70% training split.
   - Serialized preprocessor pipeline to `results/preprocessor.joblib`.
4. **Feature Synthesis:**
   - Synthesized `capital_net = capital-gain - capital-loss`.
   - Synthesized `hours_per_age = hours-per-week / (age - 16)`.
5. **Model Evaluation:** Benchmarked LogisticRegression with and without engineered features across Accuracy, F1-Score, Precision, and Recall.
6. **Label Quality Audit:** Detected duplicate feature records with conflicting target classes (label noise).

---

## 3. Results Table or Plot

### Empirical Validation: Feature Synthesis Impact on Linear Baseline

| Pipeline Formulation | Test Accuracy | F1-Score (Income >50K) | Precision | Recall | Delta / Gain |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Base Predictors (Without Feature Engineering)** | **0.8539** | **0.6624** | 0.7407 | 0.5990 | Baseline reference |
| **Synthesized Predictors (With `capital_net` & `hours_per_age`)** | **0.8566** | **0.6703** | 0.7450 | 0.6092 | **+0.0027 Acc / +0.0079 F1** |

### Quality Audit & Pipeline Integrity Log

| Quality Dimension | Metric / Finding | Engineering Resolution |
| :--- | :--- | :--- |
| **Raw Volume** | 48,842 rows x 15 columns | Ingested losslessly with zero silent drops |
| **Missing Values** | `workclass` (2,799), `occupation` (2,809), `native-country` (857) | Handled via training-split mode imputation |
| **Conflicting Duplicates** | Identical feature vectors with conflicting labels | Flagged in audit log without silent record deletion |
| **Target Skew** | ~75.9% <=50K vs ~24.1% >50K | Preserved across splits via stratified sampling |

---

## 4. Three Conclusions in Your Own Words
1. **F1-Score Prevents False Confidence on Imbalanced Data:** With a 76/24 label skew, a trivial majority-class classifier obtains 76% accuracy while achieving 0.00 F1 score. Measuring the F1-score (0.6703) is necessary to quantify true utility on the high-income minority class.
2. **Zero-Leakage Enforcement Protects Real-World Generalization:** Deriving imputation values and outlier limits strictly from the training partition prevents information leakage, ensuring test set metrics accurately reflect real-world performance.
3. **Compound Attributes Enhance Linear Decision Boundaries:** Adding domain-specific compound features (`capital_net` and `hours_per_age`) provided non-linear signals that increased baseline accuracy (+0.0027) and F1 score (+0.0079) without adding complex non-linear models.
