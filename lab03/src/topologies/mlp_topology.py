import torch
import torch.nn as nn

class UmudDenseNetwork(nn.Module):
    """
    3-Stage Fully-Connected Perceptron with Batch Normalization & Dropout
    Input: (B, 1, 28, 28) -> Flatten -> Linear(784, 240) -> BN -> ReLU -> Linear(240, 120) -> BN -> ReLU -> Linear(120, 10)
    """
    def __init__(self, in_features=784, hidden1=240, hidden2=120, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(self.flatten(x))
