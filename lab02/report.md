# Vectorized Neural Network Architecture Built from Scratch (Lab 02)
**Author:** Umud Kalbizada  
**Student ID:** 72995  
**Course:** Machine Learning Systems Architecture  

---

### 1. Research Objective & Algorithmic Scope
The objective of this laboratory exercise is to engineer a multi-layer deep perceptron entirely from foundational NumPy operations, without relying on automated differentiation frameworks like PyTorch Autograd or TensorFlow. By manually deriving and coding forward tensor activations, backpropagation error deltas, and gradient descent updates, we verify mathematical exactness and audit numerical gradient stability.

### 2. Analytical Derivations & Computational Layers
The network architecture is configured as an Affine Dense pipeline:
$$\mathbf{Z}^{[1]} = \mathbf{X} \mathbf{W}^{[1]} + \mathbf{b}^{[1]}, \quad \mathbf{A}^{[1]} = \tanh(\mathbf{Z}^{[1]})$$
$$\mathbf{Z}^{[2]} = \mathbf{A}^{[1]} \mathbf{W}^{[2]} + \mathbf{b}^{[2]}, \quad \mathbf{A}^{[2]} = \tanh(\mathbf{Z}^{[2]})$$
$$\mathbf{Z}^{[3]} = \mathbf{A}^{[2]} \mathbf{W}^{[3]} + \mathbf{b}^{[3]}, \quad \mathbf{\hat{Y}} = \text{softmax}(\mathbf{Z}^{[3]})$$

The backward pass computes loss deltas using the chain rule:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{[3]}} = \mathbf{\hat{Y}} - \mathbf{Y}_{\text{one-hot}}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[3]}} = \frac{1}{N} (\mathbf{A}^{[2]})^T \frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{[3]}}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[3]}} = \frac{1}{N} \sum_{i=1}^N \frac{\partial \mathcal{L}}{\partial \mathbf{Z}^{[3]}}$$

### 3. Numerical Verification & Experimental Convergence Matrix

#### Finite-Difference Gradient Verification ($\epsilon = 10^{-5}$)
| Parameter Matrix | Layer Shape | Analytical Gradient $\nabla_{\text{ana}}$ | Numerical Gradient $\nabla_{\text{num}}$ | Relative Error $\frac{|\nabla_{\text{ana}} - \nabla_{\text{num}}|}{\max}$ | Audit Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\mathbf{W}^{[1]}$ | $(64, 48)$ | -0.014285 | -0.014285 | $3.21 \times 10^{-7}$ | PASSED |
| $\mathbf{b}^{[1]}$ | $(1, 48)$ | 0.008412 | 0.008412 | $1.85 \times 10^{-7}$ | PASSED |
| $\mathbf{W}^{[2]}$ | $(48, 24)$ | 0.032190 | 0.032190 | $4.10 \times 10^{-7}$ | PASSED |
| $\mathbf{W}^{[3]}$ | $(24, 10)$ | -0.089124 | -0.089124 | $2.44 \times 10^{-7}$ | PASSED |

#### Hyperparameter Optimization Matrix (Digits Benchmark)
| Topology Configuration | Learning Rate ($\alpha$) | Batch Size | Epochs | Training Loss | Test Accuracy (%) | Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $64 \rightarrow 32 \rightarrow 10$ | 0.05 | 64 | 40 | 0.384 | 91.67% | 0.32 |
| $64 \rightarrow 48 \rightarrow 24 \rightarrow 10$ | 0.08 | 32 | 40 | 0.098 | 96.11% | 0.54 |
| $64 \rightarrow 64 \rightarrow 32 \rightarrow 10$ | 0.15 | 32 | 50 | 0.054 | 97.22% | 0.78 |

### 4. Conclusions & Algorithmic Insights (3 Key Points)
1. **Gradient Verification Guarantees Backprop Fidelity:** Achieving maximum relative errors below $5 \times 10^{-7}$ proves that analytical derivatives for both Tanh non-linearities and Softmax Cross-Entropy were correctly implemented without transposition bugs.
2. **Mini-Batch Vectorization Significantly Accelerates Training:** Vectorizing forward and backward passes across batches of 32 samples leverages SIMD NumPy BLAS subroutines, speeding up epoch execution by over $18\times$ compared to sample-by-sample looping.
3. **Intermediate Representation Depth Prevents Underfitting:** Moving from a single hidden layer ($64 \rightarrow 32 \rightarrow 10$) to a dual-hidden layer architecture ($64 \rightarrow 48 \rightarrow 24 \rightarrow 10$) boosted generalization accuracy from 91.67% to 96.11% on digit classification.
