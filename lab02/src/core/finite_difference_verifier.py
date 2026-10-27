import numpy as np

def verify_analytic_vs_numeric_gradients(model, X_sample, y_sample, eps=1e-5):
    """Calculates relative Frobenius error between analytical backprop and 2-sided finite difference."""
    model.forward(X_sample, y_sample)
    model.backward()
    
    max_relative_err = 0.0
    for idx, layer in enumerate([model.layer1, model.layer2, model.classifier]):
        analytic_grad = layer.grad_W.copy()
        numeric_grad = np.zeros_like(layer.weights)
        
        # Check subset of weights for fast verification
        sample_indices = np.random.RandomState(42).choice(layer.weights.size, size=min(10, layer.weights.size), replace=False)
        flat_w = layer.weights.reshape(-1)
        flat_num = numeric_grad.reshape(-1)
        flat_ana = analytic_grad.reshape(-1)
        
        for k in sample_indices:
            orig = flat_w[k]
            flat_w[k] = orig + eps
            l_plus, _ = model.forward(X_sample, y_sample)
            flat_w[k] = orig - eps
            l_minus, _ = model.forward(X_sample, y_sample)
            flat_w[k] = orig
            flat_num[k] = (l_plus - l_minus) / (2.0 * eps)
            
            diff = abs(flat_ana[k] - flat_num[k])
            scale = max(abs(flat_ana[k]), abs(flat_num[k]), 1e-8)
            rel = diff / scale
            if rel > max_relative_err:
                max_relative_err = rel
    return max_relative_err
