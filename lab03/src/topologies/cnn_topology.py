import torch
import torch.nn as nn

class UmudConvClassifier(nn.Module):
    """
    2-Stage Convolutional Hierarchy with MaxPool and Adaptive Pooling
    Input: (B, 1, 28, 28)
    Stage 1: Conv2d(1, 20, kernel_size=5) -> ReLU -> MaxPool2d(2, 2)
    Stage 2: Conv2d(20, 40, kernel_size=3) -> ReLU -> AdaptiveAvgPool2d((4, 4))
    Classifier: Linear(40*4*4, 64) -> ReLU -> Linear(64, 10)
    """
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 20, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(40 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
