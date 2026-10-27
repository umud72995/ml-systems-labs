import numpy as np

class AffineDenseLayer:
    """Fully-connected dense linear transformation with Kaiming-normal initialization."""
    def __init__(self, in_features, out_features, seed=42):
        rng = np.random.RandomState(seed)
        std = np.sqrt(2.0 / in_features)
        self.weights = rng.normal(0.0, std, (in_features, out_features))
        self.bias = np.zeros((1, out_features))
        self.grad_W = np.zeros_like(self.weights)
        self.grad_b = np.zeros_like(self.bias)
        self.cache_input = None

    def forward(self, x):
        self.cache_input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, delta_out):
        batch_size = self.cache_input.shape[0]
        self.grad_W = np.dot(self.cache_input.T, delta_out) / batch_size
        self.grad_b = np.sum(delta_out, axis=0, keepdims=True) / batch_size
        return np.dot(delta_out, self.weights.T)

    def step(self, lr):
        self.weights -= lr * self.grad_W
        self.bias -= lr * self.grad_b

class TanhActivationLayer:
    """Hyperbolic tangent non-linearity with exact analytical gradient."""
    def __init__(self):
        self.out_cache = None

    def forward(self, z):
        self.out_cache = np.tanh(z)
        return self.out_cache

    def backward(self, delta_out):
        return delta_out * (1.0 - self.out_cache ** 2)

class CrossEntropyCriterion:
    """Numerically-stabilized multinomial cross entropy loss with softmax."""
    @staticmethod
    def forward(logits, targets):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        N = logits.shape[0]
        one_hot = np.zeros_like(logits)
        one_hot[np.arange(N), targets] = 1.0
        loss = -np.sum(one_hot * np.log(np.clip(probs, 1e-12, 1.0))) / N
        return loss, probs, one_hot

    @staticmethod
    def backward(probs, one_hot):
        return probs - one_hot
