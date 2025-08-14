import torch.nn as nn
"""
    Simple 1D CNN for time-series classification.
    8 Layers 
"""
class CNN1D(nn.Module):
    def __init__(self, in_channels=3, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveMaxPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.net(x)      # (B, 64, 1)
        x = x.squeeze(-1)    # (B, 64)
        return self.classifier(x)