# Machine Learning Systems Laboratory Portfolio

**Student Researcher:** Umud Kalbizada  
**Student ID:** 72995  
**Institution:** Vistula University  
**Academic Year:** 2025/2026  
**Primary Reference:** *Machine Learning Systems* (Vijay Janapa Reddi, Harvard University)  

---

## Portfolio Architecture and Navigation

This repository implements the unified coursework curriculum for Machine Learning Systems, structured across four modular laboratory investigations adhering to strict reproducible production standards:

```
ml-systems-labs/
├── README.md               # Repository overview, roadmap, and environment specs
├── requirements.txt        # Strictly pinned dependency manifest
├── .gitignore              # Production-grade cache and artifact exclusions
├── lab01/                  # Lab 01: Systems Profiling & Environment Setup
│   ├── report.md           # Systems measurement audit & deployment tier classification
│   ├── src/                # Benchmark harness and performance instrumentation
│   └── results/            # Profiling CSVs, model binaries, and system telemetry
├── lab02/                  # Lab 02: Modular Neural Network from First Principles
│   ├── report.md           # Analytical calculus, finite-difference verification, training curves
│   ├── src/                # Vectorized NumPy tensor network and backpropagation engine
│   └── results/            # Multi-parameter convergence logs, loss plots, timing metrics
├── lab03/                  # Lab 03: Architectural Topologies and Execution Budgets
│   ├── report.md           # MLP vs CNN vs Attention trade-offs, theoretical FLOPs vs MACs
│   ├── src/                # PyTorch architecture declarations and benchmarking scripts
│   └── results/            # Pareto frontier plots, latency traces, parameter tables
└── lab04/                  # Lab 04: Industrial Data Engineering & Quality Pipeline
    ├── report.md           # Zero-leakage ETL validation, feature synthesis, label noise audit
    ├── src/                # Production pipeline orchestrator and transformation modules
    └── results/            # Data card, validation telemetry, serialized joblib artifacts
```

## Environment Reproducibility

```bash
# 1. Environment initialization
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Dependency installation
pip install -r requirements.txt
```
