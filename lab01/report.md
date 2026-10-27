# Lab 01: Systems Profiling & Environment Setup

**Student Researcher:** Umud Kalbizada (Student ID: 72995)  
**Readings:** Reddi, Chapter 1 ML Systems: Overview; Chapter 2 Systems Topics: §§2.1–2.2  

---

## 1. Goal
The primary objective of this laboratory is to establish an isolated and reproducible machine learning development environment, instantiate baseline classification topologies on the Breast Cancer Wisconsin diagnostic dataset (569 records, 30 continuous real-valued features), quantify core computational performance metrics (training time, single-sample inference latency, model storage footprints), and map each architecture against real-world hardware deployment tiers (Cloud, Mobile Edge, Microcontrollers/TinyML).

---

## 2. Method
1. **Isolated Runtime & Seed Standardization:** Configured a Python virtual environment with pinned package specifications. All stochastic operations, cross-validation boundaries, and estimators are seeded with `seed=72995`.
2. **Dataset Partitioning:** Applied stratified 70/30 train/test splitting on the 569-sample diagnostic dataset, yielding 398 training instances and 171 unseen test instances.
3. **Model Selection & Architecture:**
   - **Logistic Regression (L2 Regularized):** Parametric linear model optimizing log-loss under Ridge penalty (31 total parameters: 30 feature weights + 1 bias).
   - **Constrained Decision Tree:** Non-parametric hierarchical rule-based classifier constrained to a maximum tree depth of 5 (33 tree nodes).
4. **Systems Telemetry:**
   - **Training Time:** High-resolution wall-clock duration (`time.perf_counter()`) for model fitting.
   - **Inference Latency:** Average per-sample query duration computed across 100 repeated single-sample evaluations.
   - **Disk Footprint:** Serialized joblib artifact byte-size on disk.
5. **Deployment Tier Assessment:** Evaluated models against hardware memory constraints, compute budgets, and deterministic latency envelopes.

---

## 3. Results Table or Plot

### Empirical Systems Profiling Matrix

| Model Architecture | Parameter Count | Accuracy | F1-Score | ROC-AUC | Fit Duration (ms) | Single-Sample Latency (ms) | Storage Size (bytes) | Target Deployment Tier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (L2 Bound)** | **31** | **92.40%** | **0.9395** | **0.9847** | 106.38 ms | **0.1401 ms** | **1,103 B** | **TinyML / Ultra-Low-Power Edge** |
| **Constrained Decision Tree (depth=5)** | **33** | **92.40%** | **0.9401** | **0.9141** | 4.92 ms | **0.1124 ms** | **3,929 B** | **Microcontroller / Mobile Edge** |

### Deployment Viability Matrix

| Deployment Tier | Representative Hardware | Typical RAM / Flash | Feasibility & Design Rationale |
| :--- | :--- | :--- | :--- |
| **TinyML / MCU** | ARM Cortex-M0+ / M4 | 16–64 KB RAM, 256 KB Flash | **Highly Feasible (Logistic Regression):** With only 31 floating-point weights and 0.1401 ms per-sample query latency, this topology requires zero dynamic allocation and easily runs in bare-metal C runtime on sub-1mW microcontrollers. |
| **Mobile / Edge** | Apple A-Series, Snapdragon | 4–12 GB RAM, Multi-Core CPU | **Highly Feasible (Both Models):** The constrained decision tree executes conditional branching instructions with negligible memory overhead, fitting smoothly within on-device mobile applications. |
| **Cloud Inference** | AWS c6i / g4dn Instances | 16–64 GB RAM, Enterprise GPUs | **Viable but Economically Inefficient:** Hosting such compact models on dedicated GPU cloud clusters yields massive compute underutilization; microVMs or serverless functions are preferred. |

---

## 4. Three Conclusions in Your Own Words
1. **Linear Predictors Deliver Maximum Systems Efficiency for Tabular Medicine:** Logistic regression achieved strong diagnostic performance (92.40% accuracy, 0.9395 F1-score) while consuming only 1,103 bytes of storage and executing in 0.1401 ms per sample, outperforming complex decision boundaries in hardware-constrained environments.
2. **Computational Footprint Dictates TinyML Feasibility:** A 31-parameter linear model occupies less than 2 KB serialized on disk, making it fully deployable directly onto flash memory of microcontrollers without requiring external DRAM or specialized tensor acceleration hardware.
3. **Branching vs Linear Computational Complexity Trade-offs:** The decision tree requires zero floating-point arithmetic at inference time (only conditional threshold checks), but exhibits higher structural variance and lower generalization (92.40% accuracy) compared to the smooth linear boundary of regularized logistic regression.
