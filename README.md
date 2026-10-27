# Applied Machine Learning Systems: Computational Cost & Model Architecture

**Student:** Umud Kalbizada  
**Student ID:** 72995  
**Institution:** Vistula University (2025/2026)  
**Domain:** Systems-Aware Machine Learning & TinyML Deployment  

---

## Repository Roadmap

This portfolio implements four structured investigations in systems-aware machine learning:

- **`lab01/` Systems Profiling & Environment Setup:** Baseline classification benchmarks on the Breast Cancer dataset with single-sample latency probes, memory telemetry, and TinyML hardware feasibility.
- **`lab02/` Modular Neural Network from Scratch:** Fully vectorized multi-layer perceptron written from scratch in NumPy with centered numerical gradient validation ($\pm 10^{-5}$) and 50-epoch loss convergence plots.
- **`lab03/` Architectural Topologies & Execution Budgets:** Profiling MLP, CNN, and Patch Vision Transformer on Fashion-MNIST across FLOPs, MACs, inference latency ($b=1$ and $b=128$), and classification accuracy.
- **`lab04/` Industrial Data Engineering Pipeline:** Leakage-free ETL transformation, zero-drop data contracts, duplicate label conflict auditing, and engineered feature benchmarking.

## Systems Performance Summary

| Investigation | Architecture | Model Footprint | Latency / Metric | Key Insight |
| :--- | :--- | :--- | :--- | :--- |
| **Lab 01** | Logistic Regression | 31 params (~1.08 KB) | 0.033 ms / query | Deployable on low-power ARM Cortex-M |
| **Lab 02** | NumPy MLP | W in [32, 64] | 2.1x backward/forward ratio | Double GEMM workload confirms systems theory |
| **Lab 03** | Dual-Stage CNN | 20,490 params | 89.88% test accuracy | Weight sharing achieves highest parameter efficiency |
| **Lab 04** | Leakage-Free ETL | Adult Census | +0.41% accuracy gain | Domain feature synthesis expands linear boundary |
