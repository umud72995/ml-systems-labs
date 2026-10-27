import numpy as np
from .computational_layers import AffineDenseLayer, TanhActivationLayer, CrossEntropyCriterion

class DeepThreeLayerPerceptron:
    """3-Stage Deep Perceptron using Tanh non-linearities and Xavier/Kaiming weights."""
    def __init__(self, input_dim=64, hidden_dims=(48, 24), num_classes=10, seed=101):
        self.layer1 = AffineDenseLayer(input_dim, hidden_dims[0], seed=seed)
        self.act1 = TanhActivationLayer()
        self.layer2 = AffineDenseLayer(hidden_dims[0], hidden_dims[1], seed=seed + 1)
        self.act2 = TanhActivationLayer()
        self.classifier = AffineDenseLayer(hidden_dims[1], num_classes, seed=seed + 2)
        self.last_probs = None
        self.one_hot = None

    def forward(self, x, targets=None):
        h1 = self.act1.forward(self.layer1.forward(x))
        h2 = self.act2.forward(self.layer2.forward(h1))
        logits = self.classifier.forward(h2)

        if targets is not None:
            loss, probs, one_hot = CrossEntropyCriterion.forward(logits, targets)
            self.last_probs = probs
            self.one_hot = one_hot
            return loss, probs
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def backward(self):
        d_logits = CrossEntropyCriterion.backward(self.last_probs, self.one_hot)
        d_h2 = self.layer2.backward(self.act2.backward(self.classifier.backward(d_logits)))
        self.layer1.backward(self.act1.backward(d_h2))

    def update(self, lr):
        self.layer1.step(lr)
        self.layer2.step(lr)
        self.classifier.step(lr)
