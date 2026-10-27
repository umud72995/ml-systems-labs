# Systems Analysis of Vision Topologies & Compute Cost Trade-Offs (Lab 03)
**Author:** Umud Kalbizada  
**Student ID:** 72995  
**Institution:** Faculty of Computer Science & Engineering  
**Coursework:** Machine Learning Systems Architecture  

---

## I. Research Objectives & Problem Formulation
The overarching objective of this investigation is to empirically audit the structural cost-accuracy trade-offs inherent in modern computer vision architectures. In practical production and embedded edge deployments, algorithmic accuracy cannot be decoupled from hardware budgets—specifically arithmetic operations (FLOPs), operational parameter counts, memory footprints, and wall-clock execution latencies.

This investigation explores three fundamentally divergent paradigm classes:
1. **Dense Feedforward Multilayer Perceptron (MLP)**: Relies on unconstrained global matrix multiplications across flattened vector spaces.
2. **Convolutional Neural Network (CNN)**: Exploits localized spatial inductive biases, shift equivariance, and weight parameter sharing.
3. **Patch-Based Vision Transformer (ViT)**: Employs dynamic self-attention mechanisms to model long-range pairwise token dependencies.

By standardizing dataset ingestion across Fashion-MNIST ($28 \times 28$ grayscale single-channel images) and isolating hardware environments, we quantify the trade-offs between architectural inductive bias and operational systems cost.

---

## II. Architectural Topologies & Methodology
Each topology was engineered from foundational PyTorch primitives to highlight its unique computational dynamics:

### 1. Dense MLP Topology (`UmudDenseNetwork`)
```
Input (1x28x28) ──> Flatten (784) ──> [Linear(784, 240) + BatchNorm1d + ReLU] 
                                  ──> [Linear(240, 120) + BatchNorm1d + ReLU] 
                                  ──> Linear(120, 10) ──> Logits (10)
```
- **Structural Rationale:** Global connectivity allows fast learning but lacks 2D spatial locality. Batch normalization layers stabilize internal covariance shifts across mini-batches.

### 2. Hierarchical Convolutional Network (`UmudConvClassifier`)
```
Input (1x28x28) ──> Conv2d(1->20, k=5, p=2) ──> ReLU ──> MaxPool2d(2x2)
                ──> Conv2d(20->40, k=3, p=1) ──> ReLU ──> AdaptiveAvgPool2d(4x4)
                ──> Flatten (640) ──> Linear(640, 64) ──> ReLU ──> Linear(64, 10)
```
- **Structural Rationale:** Weight sharing across sliding kernels drastically reduces parameter overhead while capturing hierarchical edges, textures, and object parts.

### 3. Patch-Based Vision Transformer (`UmudPatchTransformer`)
```
Input (1x28x28) ──> Conv2d(1->64, k=7, s=7) ──> 16 Visual Tokens (dim=64)
                ──> Add Learnable Positional Encodings
                ──> 2x TransformerEncoderLayer (d_model=64, nhead=4, d_ff=128)
                ──> Global Mean Token Pooling ──> Linear(64, 10)
```
- **Structural Rationale:** Patches are treated as discrete linguistic tokens. Self-attention matrices evaluate dynamic affinities rather than static spatial filters.

---

## III. Empirical Performance & Systems Cost Analysis

### Systems Profile Matrix
The table below consolidates the empirical benchmarks gathered under unified test runtime configurations:

| Architectural Topology | Model Parameters | Theoretical FLOPs (M) | Inference Latency (ms) | Peak RAM Allocation (MB) | Top-1 Test Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense MLP (`UmudDenseNetwork`)** | 218,650 | 0.44 | 0.35 | 1.84 | 88.62% |
| **Hierarchical CNN (`UmudConvClassifier`)** | 50,074 | 2.58 | 0.88 | 2.45 | 91.35% |
| **Vision Transformer (`UmudPatchTransformer`)** | 71,818 | 2.12 | 1.42 | 4.80 | 85.40% |

### Latency vs. FLOPs Discrepancy (Memory-Bound vs Compute-Bound)
A critical systems takeaway is that **FLOPs do not predict inference latency in isolation**:
- The **Hierarchical CNN** executes 2.58 MFLOPs yet records an average latency of **0.88 ms**, whereas the **Vision Transformer** executes fewer operations (2.12 MFLOPs) but incurs a substantially higher latency of **1.42 ms** (+61.3% wall-clock latency).
- This divergence stems from **arithmetic intensity** and memory access latency ($Q, K, V$ matrix projections, tensor transpositions, and sequence pooling) compared to optimized direct 2D GEMM convolution kernels.

### Reconciling Profiler Output Gaps (FLOPs vs. MACs)
In literature and profiler packages (such as `thop`), a frequent source of confusion is the ratio between **MACs** (Multiply-Accumulate operations) and **FLOPs**:
$$\text{FLOPs} = 2 \times \text{MACs}$$
Each fused multiply-add constitutes two distinct floating-point operations: one multiplication and one addition. For instance, `UmudDenseNetwork` reports approximately 218k MACs, which translates to $\approx 0.44$ MFLOPs.

---

## IV. Final Deductions & Architecture Selection Recommendations (Three Key Points)

1. **Spatial Inductive Bias Maximizes Parameter Efficiency:**  
   The CNN achieved the highest classification accuracy (**91.35%**) with the fewest parameters (**50,074**)—less than one-fourth the footprint of the MLP (**218,650**). For vision workloads, translational invariance is irreplaceable when training from scratch on moderate-sized datasets.

2. **Self-Attention Topologies Suffer Without Massive Pre-Training Data:**  
   The Vision Transformer achieved an accuracy of **85.40%**, underperforming the CNN despite sophisticated multi-head attention. Because attention lacks local inductive bias, small ViTs require orders of magnitude more training samples or heavy pre-training to outperform classical convolutions.

3. **Memory Bandwidth and Kernel Dispatch Dominate Real-World Latency:**  
   Comparing computational cost solely by theoretical FLOPs is misleading for deployment on resource-constrained platforms. High tensor reshaping, token splitting, and non-contiguous memory access introduce memory bus stalls that make transformers slower per operation than standard convolutions.
