import numpy as np

class DenseLayer:
    def __init__(self, in_features, out_features, seed=42):
        rng = np.random.RandomState(seed)
        self.weights = rng.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.biases = np.zeros((1, out_features))
        self.grad_W = np.zeros_like(self.weights)
        self.grad_b = np.zeros_like(self.biases)
        self.inputs = None

    def forward(self, X):
        self.inputs = X
        return np.dot(X, self.weights) + self.biases

    def backward(self, grad_output):
        N = self.inputs.shape[0]
        self.grad_W = np.dot(self.inputs.T, grad_output) / N
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True) / N
        return np.dot(grad_output, self.weights.T)

    def update_params(self, lr):
        self.weights -= lr * self.grad_W
        self.biases -= lr * self.grad_b

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.mask = None

    def forward(self, Z):
        self.mask = (Z > 0)
        return np.where(self.mask, Z, self.alpha * Z)

    def backward(self, grad_output):
        return np.where(self.mask, grad_output, self.alpha * grad_output)

class MultilayerPerceptron:
    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        self.fc1 = DenseLayer(input_dim, hidden_dim, seed=seed)
        self.act1 = LeakyReLU(alpha=0.01)
        self.fc2 = DenseLayer(hidden_dim, hidden_dim, seed=seed+1)
        self.act2 = LeakyReLU(alpha=0.01)
        self.fc3 = DenseLayer(hidden_dim, output_dim, seed=seed+2)

    def forward(self, X, Y=None):
        out1 = self.fc1.forward(X)
        act1 = self.act1.forward(out1)
        out2 = self.fc2.forward(act1)
        act2 = self.act2.forward(out2)
        logits = self.fc3.forward(act2)

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        loss = None
        if Y is not None:
            eps = 1e-12
            loss = -np.mean(np.sum(Y * np.log(probs + eps), axis=1))

        self.probs = probs
        self.Y = Y
        return logits, probs, loss

    def backward(self):
        dZ3 = self.probs - self.Y
        dAct2 = self.fc3.backward(dZ3)
        dZ2 = self.act2.backward(dAct2)
        dAct1 = self.fc2.backward(dZ2)
        dZ1 = self.act1.backward(dAct1)
        dX = self.fc1.backward(dZ1)
        return dX

    def step(self, lr):
        self.fc1.update_params(lr)
        self.fc2.update_params(lr)
        self.fc3.update_params(lr)
