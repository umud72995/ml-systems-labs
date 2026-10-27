import os
import sys
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.neural_architecture import DeepThreeLayerPerceptron
from core.finite_difference_verifier import verify_analytic_vs_numeric_gradients

def execute_pipeline():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    net = DeepThreeLayerPerceptron(input_dim=64, hidden_dims=(48, 24), num_classes=10, seed=72995)
    
    # 1. Gradient check
    err = verify_analytic_vs_numeric_gradients(net, X_train[:32], y_train[:32])
    print(f"[VERIFY] Maximum Finite Difference Gradient Discrepancy: {err:.4e}")

    # 2. Train loop
    epochs = 40
    batch_size = 32
    n_batches = int(np.ceil(len(X_train) / batch_size))
    
    start_t = time.perf_counter()
    for ep in range(epochs):
        indices = np.random.permutation(len(X_train))
        for b in range(n_batches):
            b_idx = indices[b*batch_size : (b+1)*batch_size]
            loss, _ = net.forward(X_train[b_idx], y_train[b_idx])
            net.backward()
            net.update(lr=0.08)
    elapsed = time.perf_counter() - start_t

    # 3. Test accuracy
    test_probs = net.forward(X_test)
    preds = np.argmax(test_probs, axis=1)
    acc = np.mean(preds == y_test) * 100.0
    print(f"[RESULT] Test Accuracy: {acc:.2f}% | Training Elapsed Time: {elapsed:.3f}s")

if __name__ == "__main__":
    execute_pipeline()
